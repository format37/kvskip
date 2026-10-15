"""Stage 3: the real model, batched, where the KV fits next to the weights -- the check on
the layer-major lab's composed step times (src/lab.py), and on its KL at bs=1.

The first B stage-3 documents at exact prompt length ctx - steps (one length per batch,
the kvskip kernels need it) are prefilled as one batch into a GrowingCache; then each
mode decodes `steps` teacher-forced steps (data/docs/s3/forced.json) at batch B with CUDA
events around the model forward, as in src/run_e2e.py. With --kl (B=1) the exact fp32
probe runs first as the reference and every mode's KL is reported in multiples of the
stock-SDPA noise floor.

Output: results/e2e_s3/<ctx>_bs<B>.json

Input: as src/lab.py (stage-3 documents + data/docs/s3/forced.json), models/llama-3.1-8b.
"""

import argparse
import json

import numpy as np
import torch

import common
import kvskip_attn
from kvskip_attn import KVSKIP, GrowingCache
from lab import DENSE_HEADS, FORCED, WARMUP
from prepare_s3 import DOCS
from probe_attn import STATE

OUT = common.ROOT / "results" / "e2e_s3"
MODES = {"sdpa": ("sdpa", None), "dense": ("kvskip", "dense"), "vskip1": ("kvskip", "vskip1"),
         "refresh": ("kvskip", "refresh"), "refresh_pre": ("kvskip", "refresh_pre")}


@torch.inference_mode()
def decode(model, cache, P, dec_ids, timing):
    """dec_ids [B, S]: decode inputs (prompt's last token + forced tokens). Returns (logprobs [B, S, V] or
    None, per-step ms)."""
    extra = cache.get_seq_length() - P
    if extra > 0:
        cache.crop(-extra)
    STATE.reset()
    KVSKIP.reset()
    lps, ev = [], []
    for t in range(dec_ids.shape[1]):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        logits = model(input_ids=dec_ids[:, t:t + 1], past_key_values=cache, use_cache=True,
                       logits_to_keep=1).logits[:, -1]
        b.record()
        ev.append((a, b))
        if not timing:
            lps.append(torch.log_softmax(logits.float(), -1))
    torch.cuda.synchronize()
    ms = np.array([a.elapsed_time(b) for a, b in ev])
    return (torch.stack(lps, 1) if lps else None), ms


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--docs", nargs="+", default=list(DOCS))
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--kl", action="store_true", help="bs=1: KL vs the fp32 probe reference")
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-age", type=int, default=32)
    args = ap.parse_args()
    kvskip_attn.register()
    tok, model = common.load("8b")
    forced = json.loads(FORCED.read_text())
    docs, S = args.docs[:args.batch], args.steps
    P0 = args.ctx - S
    prompts = [common.exact_prompt(tok, common.load_named_doc(d), P0) for d in docs]
    P = P0 - 1
    dev = model.device
    pre = torch.tensor([p[:-1] for p in prompts], device=dev)
    dec = torch.tensor([[p[-1]] + forced[d]["tokens"][:S - 1] for d, p in zip(docs, prompts)], device=dev)
    model.set_attn_implementation("sdpa")
    cache = GrowingCache(capacity=P + S)
    chunk = max(256, common.PREFILL_CHUNK // len(docs))
    for s0 in range(0, P, chunk):
        model(input_ids=pre[:, s0:s0 + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
    res = {"ctx": args.ctx, "batch": len(docs), "docs": docs, "steps": S, "warmup": WARMUP, "modes": {}}
    print(f"[batch-e2e] ctx={args.ctx} bs={len(docs)} prefill={P}", flush=True)
    ref = None
    if args.kl:
        assert len(docs) == 1, "--kl needs bs=1 (the probe decode path is batch 1)"
        model.set_attn_implementation("probe")
        ref, _ = decode(model, cache, P, dec, timing=False)
    for name in args.modes:
        impl, mode = MODES[name]
        model.set_attn_implementation(impl)
        if mode:
            KVSKIP.mode, KVSKIP.tau, KVSKIP.delta, KVSKIP.max_age = mode, args.tau, args.delta, args.max_age
            KVSKIP.dense_heads, KVSKIP.dense_layers = DENSE_HEADS, (0,)
        row = {}
        if ref is not None:
            KVSKIP.count = True
            lp, _ = decode(model, cache, P, dec, timing=False)
            kl = (ref.exp() * (ref - lp)).sum(-1)
            row.update(kl_mean=float(kl.mean()), top1=float((lp.argmax(-1) == ref.argmax(-1)).float().mean()))
            if mode in ("refresh", "refresh_pre"):
                tot = kvskip_attn.log_totals()
                row["kv_read_frac_sparse_layers"] = tot["blocks_read"] / tot["blocks_total"]
            del lp
        KVSKIP.count = False
        _, ms = decode(model, cache, P, dec, timing=True)
        row.update(ms_mean=float(ms[WARMUP:].mean()), ms_median=float(np.median(ms[WARMUP:])),
                   ms_p90=float(np.quantile(ms[WARMUP:], 0.9)), ms=ms.round(3).tolist())
        res["modes"][name] = row
        print(f"  {name:12s} {row['ms_mean']:7.2f} ms/step (median {row['ms_median']:.2f})"
              + (f" KL={row['kl_mean']:.2e}" if "kl_mean" in row else ""), flush=True)
    if ref is not None and "sdpa" in res["modes"]:
        noise = res["modes"]["sdpa"]["kl_mean"]
        for row in res["modes"].values():
            row["kl_x_noise"] = row["kl_mean"] / noise
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.ctx}_bs{len(docs)}{'_' + docs[0] if len(docs) == 1 else ''}.json").write_text(
        json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
