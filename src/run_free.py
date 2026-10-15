"""Stage 4 (phase 4): free-running generation at the gate point -- does quality hold when a
mode decodes on its own tokens instead of the teacher's?

The layer-major lab (src/lab.py) cannot generate freely: it processes a decode position one
layer at a time, so the token of step t + 1 is not known when step t's layer 0 runs. Free
running needs step-major decode, i.e. the whole KV of a sequence on the card. At 64k one
Llama-3.1-8B sequence holds 8 GiB of KV; next to the 13 GiB of decoder layers it fits once
the embedding table and lm_head (1 GiB each) live on the CPU. So each of the 8 stage-3
documents runs at batch 1 (the kernels and the planner treat batch elements independently;
only the split count, i.e. the fp32 summation order, depends on the batch).

Per document: exact prompt of ctx - steps tokens (src/common.py exact_prompt), prefill into a
GrowingCache with FA2 (causal, GQA-native: HF's SDPA path repeats K/V 4x when a mask is present,
1 GiB at 64k that the card does not have), then for every condition
  1. `steps` greedy steps on the condition's own tokens (EOS suppressed),
  2. the exact fp32 reference (the probe) teacher-forced on those tokens -> the reference
     distribution on the condition's own prefixes -> KL(ref || cond) per step, and whether
     the condition's greedy pick equals the reference's on the same prefix.
The "exact" condition is the reference's own free run (its text is the exact continuation).
Stock SDPA running freely is the noise floor of this protocol.

Output: results/free/<doc>_<ctx>.json (tokens, texts, per-step KL, first divergence).

Input: the stage-3 documents (src/prepare_s3.py), models/llama-3.1-8b, the kvskip extension.
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

import common
import kvskip_attn
from kvskip_attn import KVSKIP, GrowingCache
from lab import DENSE_HEADS
from prepare_s3 import DOCS
from probe_attn import STATE
from run_collect import greedy_token

OUT = common.ROOT / "results" / "free"


def fa2_prefill_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Prefill chunks (q_len > 1) through FA2 against the cache so far, causal (bottom-right aligned);
    decode steps never come here. query [B, Hq, q, D], key / value [B, Hkv, T, D] -> [B, q, Hq, D]."""
    from flash_attn import flash_attn_func
    assert query.shape[2] > 1, "decode steps use the condition's attention"
    o = flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                        softmax_scale=scaling, causal=True)
    return o, None


class CpuEmbedding(nn.Module):
    """The embedding table on the CPU: ids -> rows on the GPU."""

    def __init__(self, emb: nn.Embedding, device):
        super().__init__()
        self.emb, self.device = emb.cpu(), device
        self.num_embeddings = emb.num_embeddings

    def forward(self, ids):
        return self.emb(ids.cpu()).to(self.device)


class CpuHead(nn.Module):
    """lm_head on the CPU in fp32: hidden [B, n, H] (GPU) -> logits [B, n, V] fp32 (GPU)."""

    def __init__(self, head: nn.Linear, device):
        super().__init__()
        self.w, self.device = head.weight.detach().float().cpu(), device

    def forward(self, h):
        return (h.float().cpu() @ self.w.T).to(self.device)


def load_model(name):
    """Decoder layers on the GPU, embedding and lm_head on the CPU."""
    kvskip_attn.register()
    import probe_attn
    probe_attn.register()
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register("fa2_prefill", fa2_prefill_attention)
    AttentionMaskInterface.register("fa2_prefill", sdpa_mask)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = common.MODELS[name]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu",
                                                 attn_implementation="sdpa").eval()
    dev = torch.device("cuda")
    emb, head = model.model.embed_tokens, model.lm_head
    model.model.embed_tokens, model.lm_head = None, None
    model.to(dev)
    model.model.embed_tokens = CpuEmbedding(emb, dev)
    model.lm_head = CpuHead(head, dev)
    return tok, model


def index_blocks(seq: torch.Tensor) -> torch.Tensor:
    """Token-index blocks of the current step: seq [T] holds the tokens so far (the current one
    last). Blocks holding an earlier copy of the current token or the token right after one."""
    T = seq.shape[0]
    same = torch.zeros(T, dtype=torch.bool, device=seq.device)
    same[:T - 1] = seq[:T - 1] == seq[T - 1]
    pos = same.clone()
    pos[1:] |= same[:T - 1]
    nb = math.ceil(T / 64)
    return torch.nn.functional.pad(pos, (0, nb * 64 - T)).view(1, nb, 64).any(-1)


@torch.inference_mode()
def decode(model, cache, P, first, steps, eos_ids, seq, forced=None):
    """`steps` decode steps from prefix length P (cache cropped back to it). forced: tokens to
    feed after `first` (teacher forcing) or None (greedy on the model's own picks). seq: a
    buffer [P + steps + 1] holding the prompt; filled with the fed tokens (token index)."""
    extra = cache.get_seq_length() - P
    if extra > 0:
        cache.crop(-extra)
    STATE.reset()
    KVSKIP.reset()
    lps, picks = [], []
    token = first
    for i in range(steps):
        seq[P + i] = token
        if KVSKIP.index_heads or KVSKIP.calib_auto_ix is not None:
            KVSKIP.idx_blocks = index_blocks(seq[:P + i + 1])
        x = torch.tensor([[token]], device="cuda")
        logits = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1].float()
        lps.append(torch.log_softmax(logits, -1).cpu())
        pick, _ = greedy_token(logits, eos_ids)
        picks.append(pick)
        token = forced[i] if forced is not None else pick
    return torch.stack(lps), picks


def set_condition(model, cond, args):
    impl, mode = {"exact": ("probe", None), "sdpa": ("sdpa", None), "dense": ("kvskip", "dense"),
                  "vskip1": ("kvskip", "vskip1"), "refresh_s3": ("kvskip", "refresh_fused"),
                  "refresh_s4": ("kvskip", "refresh_fused")}[cond]
    model.set_attn_implementation(impl)
    KVSKIP.mode = mode or "dense"
    KVSKIP.tau, KVSKIP.delta, KVSKIP.max_age, KVSKIP.local = args.tau, args.delta, args.max_age, 2
    KVSKIP.dense_layers = (0,)
    KVSKIP.index_heads, KVSKIP.index_union = (), False
    KVSKIP.calib, KVSKIP.calib_warm, KVSKIP.calib_hold, KVSKIP.calib_free = None, (), 0, False
    KVSKIP.calib_kind, KVSKIP.calib_auto_ix = "logit", None
    KVSKIP.dense_heads = ()
    if cond == "refresh_s3":                     # the stage-3 policy: 8 always-exact heads
        KVSKIP.dense_heads = DENSE_HEADS
    elif cond == "refresh_s4":                   # the stage-4 policy: online calibration, no head list
        KVSKIP.calib, KVSKIP.calib_kind = args.calib, args.calib_kind
        KVSKIP.calib_warm = tuple(args.calib_warm)
        KVSKIP.calib_auto_ix = args.calib_auto_ix
        KVSKIP.dense_layers = tuple(args.dense_layers)
    KVSKIP.count = True


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def repeats(tokens, n=8):
    """Share of n-grams that already occurred earlier in the same continuation (degeneration)."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / max(1, len(grams))


@torch.inference_mode()
def run_doc(tok, model, name, args):
    eos_ids = list(model.generation_config.eos_token_id)
    S = args.steps
    ids = common.exact_prompt(tok, common.load_named_doc(name), args.ctx - S)
    P = len(ids) - 1
    seq = torch.zeros(P + S + 1, dtype=torch.long, device="cuda")
    seq[:P] = torch.tensor(ids[:-1], device="cuda")
    t0 = time.perf_counter()
    model.set_attn_implementation("fa2_prefill")
    cache = GrowingCache(capacity=math.ceil((P + S) / 64) * 64)
    x = torch.tensor(ids[:-1], device="cuda")[None]
    for s0 in range(0, P, args.chunk):
        model(input_ids=x[:, s0:s0 + args.chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    res = {"doc": name, "ctx": args.ctx, "prompt_tokens": len(ids), "steps": S, "prefill_s": round(time.perf_counter() - t0, 1),
           "policy_s4": {"calib": args.calib, "kind": args.calib_kind, "warm": args.calib_warm,
                         "auto_ix": args.calib_auto_ix, "dense_layers": args.dense_layers, "delta": args.delta},
           "conditions": {}}
    print(f"[free] {name} ctx={args.ctx} prompt={len(ids)} prefill {res['prefill_s']}s", flush=True)
    exact_tokens = None
    for cond in ["exact"] + [c for c in args.conds if c != "exact"]:
        t0 = time.perf_counter()
        set_condition(model, cond, args)
        lp, picks = decode(model, cache, P, ids[-1], S, eos_ids, seq)
        row = {"tokens": picks, "text": tok.decode(picks), "repeat8": repeats(picks)}
        if cond in ("refresh_s3", "refresh_s4"):
            tot = kvskip_attn.log_totals()
            row["kv_read_frac_sparse_layers"] = tot["blocks_read"] / tot["blocks_total"]
            row["refresh_rate"] = tot["dense_heads"] / max(1.0, tot["masked_heads"])
        if cond == "exact":
            exact_tokens = picks
            kl = np.zeros(S)
            agree = np.ones(S)
        else:
            set_condition(model, "exact", args)
            ref, ref_picks = decode(model, cache, P, ids[-1], S, eos_ids, seq, forced=picks)
            kl = (ref.exp() * (ref - lp)).sum(-1).numpy()
            agree = (np.array(ref_picks) == np.array(picks)).astype(float)
        row.update(kl=kl.tolist(), kl_mean=float(kl.mean()), kl_p95=float(np.quantile(kl, 0.95)),
                   kl_max=float(kl.max()), top1_agree_same_prefix=float(agree.mean()),
                   first_text_diff=first_diff(picks, exact_tokens),
                   seconds=round(time.perf_counter() - t0, 1))
        res["conditions"][cond] = row
        print(f"  {cond:11s} KL={row['kl_mean']:.2e} p95={row['kl_p95']:.2e} top1(same prefix)="
              f"{row['top1_agree_same_prefix']:.3f} text differs from step {row['first_text_diff']} "
              f"rep8={row['repeat8']:.2f} {row['seconds']}s | {row['text'][:90]!r}", flush=True)
    noise = res["conditions"].get("sdpa", {}).get("kl_mean")
    if noise:
        for row in res["conditions"].values():
            row["kl_x_noise"] = row["kl_mean"] / noise
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_{args.ctx}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    del cache
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--docs", nargs="+", default=list(DOCS))
    ap.add_argument("--conds", nargs="+", default=["sdpa", "vskip1", "refresh_s3", "refresh_s4"])
    ap.add_argument("--chunk", type=int, default=1024, help="prefill chunk (tokens)")
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-age", type=int, default=32)
    ap.add_argument("--calib", type=float, default=0.05)
    ap.add_argument("--calib-kind", default="peak", choices=["logit", "add", "peak"])
    ap.add_argument("--calib-warm", nargs="*", type=int, default=[32, 8], help="W A")
    ap.add_argument("--calib-auto-ix", type=float, default=None)
    ap.add_argument("--dense-layers", nargs="*", type=int, default=[0])
    args = ap.parse_args()
    tok, model = load_model(args.model)
    for name in args.docs:
        run_doc(tok, model, name, args)


if __name__ == "__main__":
    main()
