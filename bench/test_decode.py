"""B1 correctness: kvskip vs an fp64 reference on real decode-time q/k/v (data/kv_dumps),
next to stock SDPA and FA2 on the same inputs.

Checks: dense output (bf16 and fp32 outputs, several split counts), masked
output with the dump's GQA-union top-p 0.99 masks, per-block stats, stats-only
mode, strided (FA2-layout) K/V, a batched case, D=64 and edge cases.
Pass criterion from the kickoff: kvskip bf16 error <= FA2's (~2e-3).

    .venv/bin/python bench/test_decode.py            # writes results/bench/correctness.{json,md}
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT, reference_block_stats, reference_decode, rel_err  # noqa: E402

sys.path.insert(0, str(ROOT))
from kernels import kvskip  # noqa: E402

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

DUMPS = ROOT / "data" / "kv_dumps"
OUT = ROOT / "results" / "bench"


def load_dump(path):
    with safe_open(str(path), "pt", device="cuda") as f:
        meta = f.metadata()
        layers = json.loads(meta["layers"])
        data = {l: {n: f.get_tensor(f"L{l}.{n}") for n in ("q", "k", "v", "mask099")} for l in layers}
    meta["T"] = json.loads(meta["T"])
    meta["steps"] = json.loads(meta["steps"])
    return meta, data


def sdpa(q, k, v, scale):
    return F.scaled_dot_product_attention(q[:, :, None], k, v, scale=scale, enable_gqa=True)[:, :, 0]


def fa2(q, k, v, scale):
    o = flash_attn_with_kvcache(q[:, None], k.transpose(1, 2), v.transpose(1, 2), softmax_scale=scale)
    return o[:, 0]


class Report:
    def __init__(self):
        self.rows = {}

    def add(self, name, err):
        self.rows.setdefault(name, []).append(err)

    def summary(self):
        return {k: {"n": len(v), "max": max(v), "mean": sum(v) / len(v)} for k, v in self.rows.items()}


def main():
    torch.manual_seed(0)
    rep = Report()
    stats_err = {"blk_max_abs": 0.0, "blk_sum_rel": 0.0, "weight_abs": 0.0}
    files = sorted(DUMPS.glob("8b_*_32768.safetensors"))
    assert files, "run src/dump_kv.py first"
    batch_q, batch_k, batch_v = [], [], []
    for path in files:
        meta, data = load_dump(path)
        scale = float(meta["scaling"])
        for l, d in data.items():
            for si, T in enumerate(meta["T"]):
                q = d["q"][si][None]                                  # [1, Hq, D]
                k = d["k"][None, :, :T]                               # [1, Hkv, T, D] (view)
                v = d["v"][None, :, :T]
                ref = reference_decode(q, k, v, scale)
                for ns in (None, 1, 7):
                    r = kvskip.decode(q, k, v, scale, num_splits=ns, out_f32=True)
                    tag = "auto" if ns is None else ns
                    rep.add(f"kvskip_bf16_splits_{tag}", rel_err(r["out"], ref))
                    rep.add(f"kvskip_f32_splits_{tag}", rel_err(r["out_f32"], ref))
                rep.add("sdpa_bf16", rel_err(sdpa(q, k, v, scale), ref))
                if flash_attn_with_kvcache is not None:
                    rep.add("fa2_bf16", rel_err(fa2(q, k, v, scale), ref))
                # strided K/V (FA2 memory layout [B, T, Hkv, D] seen as [B, Hkv, T, D])
                ks = k.transpose(1, 2).contiguous().transpose(1, 2)
                vs = v.transpose(1, 2).contiguous().transpose(1, 2)
                rep.add("kvskip_bf16_bshd_layout", rel_err(kvskip.decode(q, ks, vs, scale)["out"], ref))
                # stats: dense+stats and stats-only
                M, S = reference_block_stats(q, k, scale)
                r = kvskip.decode(q, k, v, scale, stats=True)
                rep.add("kvskip_bf16_with_stats", rel_err(r["out"], ref))
                fin = torch.isfinite(M)
                stats_err["blk_max_abs"] = max(stats_err["blk_max_abs"],
                                               (r["blk_max"].double() - M)[fin].abs().max().item())
                stats_err["blk_sum_rel"] = max(stats_err["blk_sum_rel"],
                                               ((r["blk_sum"].double() - S) / S)[fin].abs().max().item())
                W = torch.softmax(M + torch.log(S), -1)
                stats_err["weight_abs"] = max(stats_err["weight_abs"],
                                              (kvskip.block_weights(r["blk_max"], r["blk_sum"]).double() - W).abs().max().item())
                r2 = kvskip.decode(q, k, v, scale, stats=True, load_v=False)
                assert torch.equal(r2["blk_max"], r["blk_max"]) and torch.equal(r2["blk_sum"], r["blk_sum"]), \
                    "stats-only differs from dense+stats"
                # masked: dump's union top-p 0.99 mask at this step, plus sink and 2 newest blocks
                nb = math.ceil(T / 64)
                keep = d["mask099"][si, :, :nb].bool()[None].clone()
                keep[..., 0] = True
                keep[..., -2:] = True
                ref_m = reference_decode(q, k, v, scale, keep=keep)
                plan = kvskip.make_plan(keep)
                for ns in (None, 1, 5):
                    rm = kvskip.decode(q, k, v, scale, plan=plan, num_splits=ns, out_f32=True)
                    rep.add(f"kvskip_masked_bf16_splits_{'auto' if ns is None else ns}", rel_err(rm["out"], ref_m))
                    rep.add(f"kvskip_masked_f32_splits_{'auto' if ns is None else ns}", rel_err(rm["out_f32"], ref_m))
                rms = kvskip.decode(q, k, v, scale, plan=plan, stats=True)
                vis = keep.repeat_interleave(4, 1)                     # [1, Hq, nb]
                assert torch.isinf(rms["blk_max"][~vis]).all(), "masked stats touched a skipped block"
                assert torch.allclose(rms["blk_max"][vis], r["blk_max"][vis]), "masked stats differ"
                # two-pass V-skip (pass 1 writes P, pass 2 reads P + V of kept blocks)
                for rule, val in (("topp", 0.99), ("tau", 1e-4)):
                    r2p = kvskip.vskip_two_pass(q, k, v, scale, rule=rule, value=val, out_f32=True)
                    ref_v = reference_decode(q, k, v, scale, keep=r2p["keep"])
                    rep.add(f"vskip2_{rule}_bf16", rel_err(r2p["out"], ref_v))
                    rep.add(f"vskip2_{rule}_f32", rel_err(r2p["out_f32"], ref_v))
                    rep.add(f"vskip2_{rule}_kept_frac", r2p["kept_frac"])
                # single-pass V-skip: reference keep = the same sink-anchored rule in fp64
                for tau in (1e-4, 1e-3):
                    r1p = kvskip.vskip_one_pass(q, k, v, scale, tau=tau, out_f32=True, count=True)
                    M64, S64 = reference_block_stats(q, k, scale)
                    lse_b = M64 + torch.log(S64)                            # [1, Hq, nb]
                    keep1 = (lse_b - lse_b[..., :1] >= math.log(tau)).view(1, 8, 4, -1).any(2)
                    keep1[..., 0] = True
                    keep1[..., -2:] = True
                    ref1 = reference_decode(q, k, v, scale, keep=keep1)
                    rep.add(f"vskip1_tau{tau:g}_bf16", rel_err(r1p["out"], ref1))
                    rep.add(f"vskip1_tau{tau:g}_f32", rel_err(r1p["out_f32"], ref1))
                    rep.add(f"vskip1_tau{tau:g}_kept_frac", r1p["kept_frac"])
                    rep.add(f"vskip1_tau{tau:g}_kept_frac_ref", keep1.float().mean().item())
                    # with a normaliser estimate given (here: the exact one) = the oracle weight rule
                    lse_exact = torch.logsumexp(lse_b, -1).float().contiguous()       # [1, Hq]
                    r1e = kvskip.vskip_one_pass(q, k, v, scale, tau=tau, out_f32=True, lse_prev=lse_exact)
                    keep2 = (lse_b - lse_exact[..., None].double() >= math.log(tau)).view(1, 8, 4, -1).any(2)
                    keep2[..., 0] = True
                    keep2[..., -2:] = True
                    ref2 = reference_decode(q, k, v, scale, keep=keep2)
                    rep.add(f"vskip1_lse_tau{tau:g}_f32", rel_err(r1e["out_f32"], ref2))
        T_min = min(meta["T"])
        for l, d in data.items():
            batch_q.append(d["q"][-1])
            batch_k.append(d["k"][:, :T_min])
            batch_v.append(d["v"][:, :T_min])
        print(f"{path.name}: done", flush=True)

    # batched: B = genres x layers at a common length
    T = min(x.shape[1] for x in batch_k)
    q = torch.stack(batch_q)
    k = torch.stack([x[:, :T] for x in batch_k])
    v = torch.stack([x[:, :T] for x in batch_v])
    ref = reference_decode(q, k, v, float(meta["scaling"]))
    rb = kvskip.decode(q, k, v, float(meta["scaling"]))
    rep.add(f"kvskip_bf16_batch{q.shape[0]}", rel_err(rb["out"], ref))

    # synthetic edge cases: D=64, short / ragged T, more splits than tiles, huge score gaps
    for D, T, ns in ((64, 1000, None), (64, 33, 3), (128, 64, 4), (128, 65, 9), (128, 3000, 64)):
        q = torch.randn(2, 32, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, 8, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(2, 8, T, D, device="cuda", dtype=torch.bfloat16)
        k[:, :, 0] *= 8.0          # a sink with a ~30-nat gap over the rest
        scale = D ** -0.5
        ref = reference_decode(q, k, v, scale)
        rep.add(f"synthetic_D{D}_T{T}_splits{ns}", rel_err(kvskip.decode(q, k, v, scale, num_splits=ns)["out"], ref))

    summ = rep.summary()
    res = {"cases": summ, "stats": stats_err}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "correctness.json").write_text(json.dumps(res, indent=1))
    lines = ["| check | cases | max rel. error | mean rel. error |", "|---|---|---|---|"]
    # (rows named *_kept_frac hold the kept share of blocks, not an error)
    for name, s in summ.items():
        lines.append(f"| {name} | {s['n']} | {s['max']:.2e} | {s['mean']:.2e} |")
    lines.append("")
    lines.append(f"Block stats vs fp64: max |Δ block max| = {stats_err['blk_max_abs']:.2e} nats, "
                 f"max rel. Δ block sumexp = {stats_err['blk_sum_rel']:.2e}, "
                 f"max |Δ block weight| = {stats_err['weight_abs']:.2e}.")
    (OUT / "correctness.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fa = summ.get("fa2_bf16")
    ours = summ["kvskip_bf16_splits_auto"]
    if fa:
        print(f"\nkvskip bf16 max err {ours['max']:.2e} vs FA2 {fa['max']:.2e}: "
              f"{'PASS' if ours['max'] <= fa['max'] * 1.05 else 'FAIL'}")


if __name__ == "__main__":
    main()
