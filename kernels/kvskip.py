"""Python front-end of the kvskip decode kernels.

The extension is built on first use with torch.utils.cpp_extension.load
(ninja, sm_89) into kernels/build/.

    out = decode(q, k, v, scale)                         # dense
    out = decode(q, k, v, scale, plan=make_plan(keep))   # skip blocks (neither K nor V read)
    out = decode(q, k, v, scale, stats=True)             # dense + per-block max / sumexp
    out = decode(q, k, v, scale, stats=True, load_v=False)  # stats only: reads K, not V

q [B, Hq, D] or [B, Hq, 1, D]; k, v [B, Hkv, T, D] in any batch/head/token
strides with D contiguous (HF DynamicCache layout, or FA2's [B, T, Hkv, D]
passed as k.transpose(1, 2)). Blocks are 64 tokens from position 0.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TILE = 64
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        build = HERE / "build"
        build.mkdir(exist_ok=True)
        verbose = bool(os.environ.get("KVSKIP_VERBOSE"))
        cuda_flags = ["-O3", "-lineinfo", "--use_fast_math"]
        if verbose:
            cuda_flags.append("-Xptxas=-v")
        _EXT = load(name="kvskip_ext",
                    sources=[str(HERE / "kvskip_ext.cpp"), str(HERE / "kvskip_kernel.cu")],
                    extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags,
                    build_directory=str(build), verbose=verbose)
    return _EXT


def num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def num_splits_for(bh: int, n_tiles: int, sms: int | None = None, max_splits: int = 64,
                   min_tiles: int = 4) -> int:
    """Split count for B*Hkv work items of n_tiles tiles each (one CTA per SM).

    Picks the best wave efficiency (CTAs / (waves * SMs)), lightly preferring
    fewer splits; every split keeps at least `min_tiles` tiles."""
    sms = sms or num_sms()
    best, best_score = 1, -1.0
    for s in range(1, max_splits + 1):
        if s > 1 and math.ceil(n_tiles / s) < min_tiles:
            break
        waves = bh * s / sms
        eff = waves / math.ceil(waves)
        score = eff - 0.002 * s
        if score > best_score + 1e-9:
            best, best_score = s, score
    return best


@dataclass
class Plan:
    """Kept-block list for the masked kernel (built once per mask)."""
    block_idx: torch.Tensor   # int32 [B, Hkv, nb], kept blocks first, ascending
    block_cnt: torch.Tensor   # int32 [B, Hkv]
    max_cnt: int              # max kept blocks over (b, kv head), sizes the split count
    nb: int


def make_plan(keep: torch.Tensor, sync: bool = True) -> Plan:
    """keep: bool [B, Hkv, nb] -> Plan. With sync=True one host sync sizes the split
    count by the largest kept list; sync=False sizes it by nb (no sync, over-split)."""
    cnt = keep.sum(-1, dtype=torch.int32)
    order = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return Plan(order.contiguous(), cnt.contiguous(), int(cnt.max()) if sync else keep.shape[-1],
                keep.shape[-1])


def decode(q, k, v, scale: float, *, plan: Plan | None = None, stats: bool = False, load_v: bool = True,
           num_splits: int | None = None, stages: int = 2, out_f32: bool = False, write_p: bool = False) -> dict:
    """Decode attention for one layer. Returns dict(out [B, Hq, D] bf16, lse [B, Hq] fp32,
    blk_max / blk_sum [B, Hq, nb] fp32 (stats), out_f32 (if asked), p (write_p: fp16
    [B, Hq, nb*64] = exp(s - blk_max), the input of the two-pass V-skip's second pass))."""
    if q.dim() == 4:
        q = q.squeeze(2)
    B, Hq, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    nb = math.ceil(T / TILE)
    n_items = nb if plan is None else plan.max_cnt
    if num_splits is None:
        num_splits = num_splits_for(B * Hkv, n_items)
    res = ext().decode(q, k, v, float(scale),
                       None if plan is None else plan.block_idx,
                       None if plan is None else plan.block_cnt,
                       stats, load_v, int(num_splits), int(stages), out_f32, write_p)
    out, lse, bmax, bsum, o32, pbuf = res
    d = {"out": out if load_v else None, "lse": lse, "num_splits": num_splits}
    if stats:
        d["blk_max"], d["blk_sum"] = bmax, bsum
    if out_f32:
        d["out_f32"] = o32
    if write_p:
        d["p"] = pbuf
    return d


def block_weights(blk_max: torch.Tensor, blk_sum: torch.Tensor) -> torch.Tensor:
    """Per-block softmax mass [B, Hq, nb] from the stats outputs."""
    logw = blk_max + torch.log(blk_sum)
    return torch.softmax(logw, dim=-1)


def always_keep(nb: int, local: int, device) -> torch.Tensor:
    """Blocks every sparse mode reads: the sink block and the `local` newest blocks."""
    a = torch.zeros(nb, dtype=torch.bool, device=device)
    a[0] = True
    a[-local:] = True
    return a


def select_blocks(logw: torch.Tensor, G: int, rule: str, value: float, local: int = 2) -> torch.Tensor:
    """GQA-union block selection from per-head block log-weights [B, Hq, nb] (unnormalised).

    rule "tau": keep a block if any head of the group has weight >= value;
    rule "topp": keep the fewest blocks holding >= value of each head's mass (union over heads).
    Returns keep [B, Hkv, nb] bool (sink + `local` newest blocks always kept)."""
    B, Hq, nb = logw.shape
    W = torch.softmax(logw, dim=-1)
    if rule == "tau":
        keep_h = W >= value
    elif rule == "topp":
        ws, order = W.sort(-1, descending=True)
        take = (ws.cumsum(-1) - ws) < value
        keep_h = torch.zeros_like(take).scatter(-1, order, take)
    else:
        raise ValueError(rule)
    keep = keep_h.view(B, Hq // G, G, nb).any(2)
    return keep | always_keep(nb, local, logw.device)


def pv(P, v, blk_scale, plan: Plan, num_splits: int | None = None, out_f32: bool = False):
    B, Hkv = v.shape[0], v.shape[1]
    if num_splits is None:
        num_splits = num_splits_for(B * Hkv, plan.max_cnt)
    return ext().pv(P, v, blk_scale, plan.block_idx, plan.block_cnt, int(num_splits), out_f32)


def vskip_two_pass(q, k, v, scale: float, rule: str = "topp", value: float = 0.99, local: int = 2,
                   sync: bool = True, out_f32: bool = False) -> dict:
    """Two-pass V-skip: pass 1 reads all of K once and writes block stats and P (fp16);
    the planner picks blocks per GQA union; pass 2 reads P and V of the kept blocks only.
    Bytes vs dense: K + P(write) + kept * (V + P(read)) instead of K + V."""
    if q.dim() == 4:
        q = q.squeeze(2)
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    r1 = decode(q, k, v, scale, stats=True, load_v=False, write_p=True)
    logw = r1["blk_max"] + torch.log(r1["blk_sum"])                    # [B, Hq, nb]
    keep = select_blocks(logw, G, rule, value, local)                   # [B, Hkv, nb]
    keep_h = keep.repeat_interleave(G, dim=1)                           # [B, Hq, nb]
    lse_kept = logw.masked_fill(~keep_h, float("-inf")).logsumexp(-1, keepdim=True)
    scale_b = torch.exp(r1["blk_max"] - lse_kept).masked_fill(~keep_h, 0.0)
    scale_b = scale_b.view(B, Hkv, G, -1).transpose(2, 3).contiguous()  # [B, Hkv, nb, G]
    plan = make_plan(keep, sync=sync)
    out, o32 = pv(r1["p"], v, scale_b, plan, out_f32=out_f32)
    d = {"out": out, "keep": keep, "kept_frac": keep.float().mean().item() if sync else None}
    if out_f32:
        d["out_f32"] = o32
    return d


def vskip_one_pass(q, k, v, scale: float, tau: float = 1e-4, local: int = 2, num_splits: int | None = None,
                   out_f32: bool = False, count: bool = False, lse_prev: torch.Tensor | None = None) -> dict:
    """Single-pass V-skip: K read once, V of a block read only if some head of the
    group has estimated weight >= tau there. The normaliser estimate is lse_prev
    (the previous decode step's log-sum-exp per query head, e.g. this call's "lse"
    output one step earlier) if given; otherwise every CTA bounds it from below
    with the sink block's log-sum-exp, which makes the rule conservative (it keeps
    every block the oracle "weight >= tau" rule keeps) but skips less."""
    if q.dim() == 4:
        q = q.squeeze(2)
    B, Hq, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    nb = math.ceil(T / TILE)
    if num_splits is None:
        num_splits = num_splits_for(B * Hkv, nb)
    out, lse, o32, cnt = ext().vskip1(q, k, v, float(scale), float(tau), int(local), int(num_splits),
                                      out_f32, count, lse_prev)
    d = {"out": out, "lse": lse, "num_splits": num_splits}
    if out_f32:
        d["out_f32"] = o32
    if count:
        d["v_tiles"] = cnt
        d["kept_frac"] = (cnt.sum() / (B * Hkv * nb)).item()
    return d
