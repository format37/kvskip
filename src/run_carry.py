"""Stage 5 (phase 2): the calibration state across requests.

Stage 4's calibration (a peak-hold correction c per KV head, a warm-up, the automatic token index)
starts from zero on every sequence, so a head that misses about once per document (Llama's (5, 2) on
code at 32k) cannot be caught before its first miss. Here the documents run one after another in a
given order, and every condition with `carry` hands its state to the next document
(src/reuse.py, ReuseConfig.carry):
  end    c starts at carry * c at the end of the previous document; heads above the index threshold
         start latched onto the token index (the kickoff's variant);
  latch  heads latched onto the token index stay latched, c restarts at 0;
  floor  a slow memory m = max(carry * m, the document's worst residual real - estimated on its own
         refresh steps) starts c and stays its floor for the whole document (a head that once missed
         stays as sensitive as its worst miss, fading by `carry` per document).
A condition without carry is stage 4 (every document from zero); `cw=0` drops the warm-up once a
state is carried.

Per document (the same protocol as src/run_reuse.py): prefill, the exact fp32 reference greedy run,
stock bf16 SDPA teacher-forced (noise floor), then every condition teacher-forced on the reference's
tokens -> KL x noise, the worst step, K+V bytes, and per KV head: c at the end, the document's worst
residual (c_peak), latched or not, refresh rate, worst coverage.

Output: results/carry/<model>_<ctx>_<order>[_<tag>].{json,heads.parquet,steps.parquet}; with --trace also
.trace.parquet: every step of the traced heads (refresh, coverage, estimate, correction, latch, age).
Input: data/docs/{,a4/,s3/}<name>.json, models/<model>.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
import run_reuse
from probe_attn import STATE
from run_reuse import REC_KEYS, bytes_of, kl, parse_cond, vocab_classes

OUT = common.ROOT / "results" / "carry"


def seed(cfg, state, L, device):
    """Hand the carried state of cfg's chain (None on the first document) to the condition."""
    cfg.calib_init = cfg.latch_init = cfg.calib_floor = None
    if state is None:
        return
    as_dict = lambda a: {l: torch.tensor(a[l], device=device) for l in range(L)}
    if cfg.carry_kind == "end":
        cfg.calib_init = as_dict(cfg.carry * state["c_end"])
    elif cfg.carry_kind == "latch":
        cfg.latch_init = as_dict(state["latched"])
    elif cfg.carry_kind == "floor":
        cfg.calib_init = as_dict(state["m"])
        cfg.calib_floor = as_dict(state["m"])
    else:
        raise ValueError(cfg.carry_kind)


def advance(cfg, state, c_end, peak, latched):
    """The state the next document starts from."""
    if cfg.carry_kind == "latch":
        return {"latched": latched | (False if state is None else state["latched"])}
    if cfg.carry_kind == "floor":
        m = peak if state is None else np.maximum(cfg.carry * state["m"], peak)
        return {"m": m}
    return {"c_end": c_end}


TRACE_KEYS = ("refresh", "covered_min", "covered_stale_min", "miss_p", "calib", "auto_ix", "age", "idx_cov", "m_after")


def run_order(tok, model, names, ctx, steps, conds, order, trace=()):
    L = model.config.num_hidden_layers
    Hkv = model.config.num_key_value_heads
    D = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    G = model.config.num_attention_heads // Hkv
    eos_ids = common.eos_ids(model)
    configs = [parse_cond(c) for c in conds]
    chain = {cfg.name(): None for cfg in configs}
    docs, head_rows, step_rows, trace_rows = [], [], [], []
    for pos, name in enumerate(names):
        ids, _ = common.build_prompt(tok, common.load_named_doc(name), ctx - steps)
        prefix_len = len(ids) - 1
        run_reuse.IDS = torch.zeros(prefix_len + steps, dtype=torch.long, device=model.device)
        run_reuse.IDS[:prefix_len] = torch.tensor(ids[:-1], device=model.device)
        cache = common.prefill(model, ids[:-1])
        first = ids[-1]
        t0 = time.perf_counter()
        ref_lp, ref_tokens, _ = run_reuse.decode(model, cache, prefix_len, first, steps, eos_ids)
        STATE.passthrough = "sdpa"
        noise_lp, _, _ = run_reuse.decode(model, cache, prefix_len, first, steps, eos_ids, ref_tokens)
        STATE.passthrough = None
        noise_kl = kl(ref_lp, noise_lp).cpu().numpy()
        print(f"[carry] {order} #{pos + 1} {name} ctx={ctx}: prompt={len(ids)} ref+noise "
              f"{time.perf_counter() - t0:.0f}s, noise KL {noise_kl.mean():.2e}", flush=True)
        doc = {"doc": name, "position": pos + 1, "noise_kl_mean": float(noise_kl.mean()), "conditions": {}}
        for cfg in configs:
            key = cfg.name()
            seed(cfg, chain[key], L, model.device)
            t0 = time.perf_counter()
            lp, toks, recs = run_reuse.decode(model, cache, prefix_len, first, steps, eos_ids, ref_tokens, cfg)
            k = kl(ref_lp, lp).cpu().numpy()
            R = {r: torch.stack([x[r] for x in recs]).cpu().numpy() for r in REC_KEYS}     # [S, L, Hkv]
            sparse = [l for l in range(L) if l not in cfg.dense_layers]
            b = bytes_of(cfg, R["kept"], R["kept"], L, D, G)
            c_end = R["calib"][-1]
            # this document's own worst invisible miss: real - estimated on its refresh steps (what the peak-hold
            # correction samples), without the value the condition started from
            resid = np.where(R["refresh"][1:] > 0, 1.0 - R["covered_stale_min"][1:] - R["miss_p"][1:], 0.0)
            peak = np.clip(resid.max(0), 0.0, None)
            latched = R["auto_ix"][-1] > 0
            eff_end = c_end if cfg.calib_floor is None else np.maximum(
                c_end, np.stack([cfg.calib_floor[l].cpu().numpy() for l in range(L)]))
            if cfg.carry is not None:
                chain[key] = advance(cfg, chain[key], c_end, peak, latched)
            worst = int(np.argmax(k))
            doc["conditions"][key] = {
                **b, "kl_x_noise": float(k.mean() / noise_kl.mean()), "kl_mean": float(k.mean()),
                "worst_step": worst, "worst_step_x_noise": float(k[worst] / noise_kl.mean()),
                "kl_x_noise_wo_worst3": float(np.sort(k)[:-3].mean() / noise_kl.mean()),
                "refresh_rate": float(R["refresh"][1:, sparse].mean()),
                "latched_start": int((R["auto_ix"][0][sparse] > 0).sum()),
                "latched_end": int(latched[sparse].sum()),
                "heads_c_end_gt_0.05": int((eff_end[sparse] > 0.05).sum()),
                "heads_peak_gt_0.1": int((peak[sparse] > 0.1).sum()),
                "seconds": round(time.perf_counter() - t0, 1)}
            c = doc["conditions"][key]
            print(f"  {key:52s} KL x{c['kl_x_noise']:.2f} worst step {worst} x{c['worst_step_x_noise']:.0f} "
                  f"bytes={c['bytes_total']:.3f} refresh={c['refresh_rate']:.3f} latched {c['latched_start']}"
                  f"->{c['latched_end']} c>0.05: {c['heads_c_end_gt_0.05']} {c['seconds']}s", flush=True)
            cov = R["covered_min"].min(0)
            for l in sparse:
                for h in range(Hkv):
                    head_rows.append({"order": order, "position": pos + 1, "doc": name, "condition": key,
                                      "layer": l, "kv_head": h, "c_end": float(c_end[l, h]),
                                      "c_eff_end": float(eff_end[l, h]), "c_peak": float(peak[l, h]),
                                      "latched": bool(latched[l, h]),
                                      "refresh_rate": float(R["refresh"][1:, l, h].mean()),
                                      "covered_min": float(cov[l, h])})
            step_rows += [{"order": order, "position": pos + 1, "doc": name, "condition": key, "step": i,
                           "kl_x_noise": float(k[i] / noise_kl.mean())} for i in range(steps)]
            for l, h in trace:
                trace_rows += [{"order": order, "position": pos + 1, "doc": name, "condition": key, "layer": l,
                                "kv_head": h, "step": i, "kl_x_noise": float(k[i] / noise_kl.mean()),
                                "floor": float(cfg.calib_floor[l][h]) if cfg.calib_floor is not None else 0.0,
                                **{t: float(R[t][i, l, h]) for t in TRACE_KEYS}} for i in range(steps)]
        docs.append(doc)
        del cache
        torch.cuda.empty_cache()
    return docs, head_rows, step_rows, trace_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=list(common.MODELS))
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--docs", nargs="+", required=True, help="document names in the forward order")
    ap.add_argument("--orders", nargs="+", default=["fwd", "rev"], choices=["fwd", "rev"])
    ap.add_argument("--cond", nargs="+", required=True, help="conditions, see run_reuse.parse_cond")
    ap.add_argument("--trace", nargs="*", default=[], help="heads L.H to record step by step")
    ap.add_argument("--tag", default="", help="output name suffix")
    args = ap.parse_args()
    trace = [tuple(int(x) for x in t.split(".")) for t in args.trace]
    tok, model = common.load(args.model)
    STATE.vocab = vocab_classes(tok, model.device, model.config.vocab_size)
    OUT.mkdir(parents=True, exist_ok=True)
    for order in args.orders:
        names = args.docs if order == "fwd" else args.docs[::-1]
        docs, heads, steps, traced = run_order(tok, model, names, args.ctx, args.steps, args.cond, order, trace)
        stem = OUT / (f"{args.model}_{args.ctx}_{order}" + (f"_{args.tag}" if args.tag else ""))
        Path(f"{stem}.json").write_text(json.dumps({"model": args.model, "ctx": args.ctx, "order": order,
                                                    "docs": docs}, indent=1))
        pd.DataFrame(heads).to_parquet(f"{stem}.heads.parquet")
        pd.DataFrame(steps).to_parquet(f"{stem}.steps.parquet")
        if traced:
            pd.DataFrame(traced).to_parquet(f"{stem}.trace.parquet")


if __name__ == "__main__":
    main()
