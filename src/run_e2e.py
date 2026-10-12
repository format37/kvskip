"""Stage 2: the kvskip kernels inside Llama-3.1-8B, end to end.

Same protocol as A1 (src/run_reuse.py): prompt of ctx - steps tokens, exact fp32
probe reference greedy run, then each condition teacher-forced on the reference
tokens -> KL(ref || cond) per step, top-1 agreement. The attention of the
conditions runs on the CUDA kernels (src/kvskip_attn.py), so this checks that
the kernels + planner reproduce what the probe simulated in A1/A2:
  kvskip_dense     vs stock bf16 SDPA (noise floor)
  kvskip_vskip2    vs probe "fresh"          (top-p 0.99 union, K read in full)
  kvskip_vskip1    vs probe "vprev" / "vsink" tau (single pass, previous-step normaliser / sink bound)
  kvskip_refresh   vs probe "estd" delta+age (K+V skip with the centroid trigger)
With --timing it also measures the wall time of a decode step (CUDA events,
GrowingCache, so no per-step torch.cat) for SDPA and each kvskip mode.

Output: results/e2e/<model>_<genre>_<ctx>.json
"""

import argparse
import json
import time

import numpy as np
import torch

import common
import kvskip_attn
from kvskip_attn import KVSKIP, GrowingCache
from probe_attn import STATE
from run_collect import greedy_token

OUT = common.ROOT / "results" / "e2e"


@torch.inference_mode()
def prefill(model, ids):
    cache = GrowingCache()
    x = torch.tensor(ids, device=model.device)[None]
    for s in range(0, x.shape[1], common.PREFILL_CHUNK):
        model(input_ids=x[:, s:s + common.PREFILL_CHUNK], past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


@torch.inference_mode()
def decode(model, cache, prefix_len, first, steps, eos_ids, forced=None, timing=False):
    extra = cache.get_seq_length() - prefix_len
    if extra > 0:
        cache.crop(-extra)
    STATE.reset()
    KVSKIP.reset()
    lps, toks, times = [], [], []
    token = first
    for i in range(steps):
        if timing:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
        logits = common.step(model, token, cache)
        if timing:
            b.record()
            times.append((a, b))
        lps.append(torch.log_softmax(logits, -1))
        pick, _ = greedy_token(logits, eos_ids)
        toks.append(pick)
        token = forced[i] if forced is not None else pick
    torch.cuda.synchronize()
    ms = [a.elapsed_time(b) for a, b in times]
    return torch.stack(lps), toks, ms


def kl(a, b):
    return (a.exp() * (a - b)).sum(-1)


def run(model_name, genre, ctx, steps, tok, model, args):
    eos_ids = list(model.generation_config.eos_token_id)
    ids, _ = common.build_prompt(tok, common.load_doc(genre), ctx - steps)
    prefix_len = len(ids) - 1
    model.set_attn_implementation("probe")
    cache = prefill(model, ids[:-1])
    first = ids[-1]
    ref_lp, ref_tok, _ = decode(model, cache, prefix_len, first, steps, eos_ids)
    res = {"model": model_name, "genre": genre, "ctx": ctx, "steps": steps, "prompt_tokens": len(ids),
           "conditions": {}}

    tag = f"delta{args.delta:g}_age{args.max_age}_dh{len(args.dense_heads)}"
    conds = [("stock_sdpa_bf16", "sdpa", None), ("kvskip_dense", "kvskip", "dense"),
             ("kvskip_vskip2_topp0.99", "kvskip", "vskip2"),
             (f"kvskip_vskip1_prev_tau{args.tau:g}", "kvskip", "vskip1"),
             (f"kvskip_vskip1_sink_tau{args.tau:g}", "kvskip", "vskip1sink"),
             (f"kvskip_refresh_{tag}", "kvskip", "refresh"),
             (f"kvskip_refresh_pre_{tag}", "kvskip", "refresh_pre"),       # stage 3: trigger before the pass
             (f"kvskip_refresh_fused_{tag}", "kvskip", "refresh_fused")]   # stage 3: planner in kernels
    conds = [c for c in conds if c[1] == "sdpa" or c[2] in args.modes]
    for name, impl, mode in conds:
        model.set_attn_implementation(impl)
        if mode:
            KVSKIP.mode, KVSKIP.tau, KVSKIP.delta, KVSKIP.max_age = mode, args.tau, args.delta, args.max_age
            KVSKIP.dense_heads = tuple(tuple(int(x) for x in lh.split(".")) for lh in args.dense_heads)
        t0 = time.perf_counter()
        lp, toks, _ = decode(model, cache, prefix_len, first, steps, eos_ids, ref_tok)
        k = kl(ref_lp, lp).cpu().numpy()
        match = np.array(toks) == np.array(ref_tok)
        row = {"kl_mean": float(k.mean()), "kl_p95": float(np.quantile(k, 0.95)), "kl_max": float(k.max()),
               "top1_agree": float(match.mean()), "seconds": round(time.perf_counter() - t0, 1)}
        if mode in ("refresh", "refresh_pre", "refresh_fused"):
            tot = kvskip_attn.log_totals()
            row.update(kv_read_frac_sparse_layers=tot["blocks_read"] / tot["blocks_total"],
                       refresh_rate=tot["dense_heads"] / max(1, tot["masked_heads"]))
        res["conditions"][name] = row
        print(f"  {name:34s} KL={row['kl_mean']:.2e} p95={row['kl_p95']:.2e} top1={row['top1_agree']:.3f}"
              + (f" KV read={row['kv_read_frac_sparse_layers']:.3f} refresh={row['refresh_rate']:.3f}"
                 if "kv_read_frac_sparse_layers" in row else "") + f" ({row['seconds']}s)", flush=True)
    noise = res["conditions"]["stock_sdpa_bf16"]["kl_mean"]
    for r in res["conditions"].values():
        r["kl_x_noise"] = r["kl_mean"] / noise

    if args.timing:
        tm = {}
        for name, impl, mode in conds:
            model.set_attn_implementation(impl)
            if mode:
                KVSKIP.mode = mode
            _, _, ms = decode(model, cache, prefix_len, first, args.timing_steps, eos_ids, ref_tok, timing=True)
            tm[name] = {"ms_median": float(np.median(ms[4:])), "ms_p90": float(np.quantile(ms[4:], 0.9))}
            print(f"  timing {name:34s} {tm[name]['ms_median']:.2f} ms/step", flush=True)
        res["timing"] = tm
    model.set_attn_implementation("probe")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model_name}_{genre}_{ctx}{args.tag}.json").write_text(json.dumps(res, indent=1))
    del cache
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=list(common.MODELS))
    ap.add_argument("--genre", nargs="+", default=list(common.GENRES))
    ap.add_argument("--ctx", nargs="+", type=int, default=[32768])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-age", type=int, default=32)
    ap.add_argument("--dense-heads", nargs="*", default=[], help="always-exact KV heads as layer.head")
    ap.add_argument("--modes", nargs="+", default=["dense", "vskip2", "vskip1", "vskip1sink", "refresh"],
                    help="kvskip modes to run (stock SDPA always runs: it is the noise floor)")
    ap.add_argument("--tag", default="", help="output suffix")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--timing-steps", type=int, default=32)
    args = ap.parse_args()
    kvskip_attn.register()
    tok, model = common.load(args.model)
    for ctx in args.ctx:
        for genre in args.genre:
            print(f"[e2e] {args.model} {genre} ctx={ctx}", flush=True)
            run(args.model, genre, ctx, args.steps, tok, model, args)


if __name__ == "__main__":
    main()
