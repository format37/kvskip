"""Stage 2 / A1 + A2: block-selection reuse ablation (see src/reuse.py).

For every document: one prefill (shared), one exact fp32 reference greedy run
(which also accumulates the A2 locality statistics), the stock bf16 SDPA noise
floor, then each reuse condition teacher-forced on the reference tokens ->
KL(ref || cond) per step, top-1 agreement, bytes skipped and per
(step, layer, KV head) records of coverage / refresh / query drift.

Outputs results/reuse/<model>_<genre>_<ctx>[.<grid>].{steps.parquet,heads.parquet,summary.json}
and <model>_<genre>_<ctx>.locality.npz (reference pass only).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
import reuse
from probe_attn import STATE
from reuse import ReuseConfig
from run_collect import greedy_token

OUT = common.ROOT / "results" / "reuse"
REC_KEYS = ("covered_min", "covered_mean", "covered_stale_min", "kept", "refresh", "drift", "age",
            "miss_c", "miss_d")


def parse_cond(spec: str) -> ReuseConfig:
    """mode[:value[:max_age]][:masked][:dh=L.H,L.H...], e.g. fresh, timer:16, oracle:0.98:32,
    estd:0.02:32, timer:16:masked, estd:0.02:32:dh=2.5,25.3 (always-exact KV heads)."""
    parts = spec.split(":")
    mode = parts[0]
    extra = {}
    if parts[-1].startswith("dh="):
        extra["dense_heads"] = tuple(tuple(int(x) for x in lh.split(".")) for lh in parts[-1][3:].split(","))
        parts = parts[:-1]
    if mode == "fresh":
        return ReuseConfig("fresh", **extra)
    if parts[-1] == "masked":
        extra["refresh"] = "masked"
        parts = parts[:-1]
    if len(parts) > 2:
        extra["max_age"] = int(parts[2])
    if mode == "timer":
        return ReuseConfig("timer", N=int(parts[1]), **extra)
    if mode == "oracle":
        return ReuseConfig("oracle", theta=float(parts[1]), **extra)
    return ReuseConfig(mode, delta=float(parts[1]), **extra)


GRIDS = {
    "main": ["fresh", "timer:4", "timer:8", "timer:16", "timer:32", "oracle:0.98", "oracle:0.95"],
    "diag": ["timer:32", "vtau:0.0001", "vsink:0.0001", "vprev:0.0001", "vprev2:0.0001"],
}


def grid(name: str, conds=()) -> list[ReuseConfig]:
    return [parse_cond(c) for c in (conds or GRIDS[name])]


def kl(ref_lp, lp):
    return (ref_lp.exp() * (ref_lp - lp)).sum(-1)


def decode(model, cache, prefix_len, first, steps, eos_ids, forced=None, cfg=None):
    extra = cache.get_seq_length() - prefix_len
    if extra > 0:
        cache.crop(-extra)
    STATE.reset()
    STATE.reuse = cfg
    L = model.config.num_hidden_layers
    lps, tokens, recs = [], [], []
    token = first
    for i in range(steps):
        STATE.t = i
        logits = common.step(model, token, cache)
        lps.append(torch.log_softmax(logits, -1))
        if cfg is not None:
            recs.append({k: torch.stack([STATE.reuse_rec[l][k].float() for l in range(L)]) for k in REC_KEYS})
        pick, _ = greedy_token(logits, eos_ids)
        tokens.append(pick)
        token = forced[i] if forced is not None else pick
    STATE.reuse = None
    return torch.stack(lps), tokens, recs


def run(model_name, genre, ctx, steps, tok, model, configs, grid_name, with_ref_stats, doc=None):
    """doc: optional document dict (data/docs format) used instead of data/docs/<genre>.json;
    `genre` then only names the outputs."""
    eos_ids = list(model.generation_config.eos_token_id)
    ids, _ = common.build_prompt(tok, doc if doc is not None else common.load_doc(genre), ctx - steps)
    prefix_len = len(ids) - 1
    cache = common.prefill(model, ids[:-1])
    first = ids[-1]
    L = model.config.num_hidden_layers
    stem = OUT / f"{model_name}_{genre}_{ctx}"
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if with_ref_stats:
        STATE.locality = reuse.Locality(L, model.device)
    ref_lp, ref_tokens, _ = decode(model, cache, prefix_len, first, steps, eos_ids)
    if with_ref_stats:
        np.savez_compressed(f"{stem}.locality.npz", **STATE.locality.to_numpy())
        STATE.locality = None
    t_ref = time.perf_counter() - t0

    STATE.passthrough = "sdpa"
    noise_lp, noise_tok, _ = decode(model, cache, prefix_len, first, steps, eos_ids, ref_tokens)
    STATE.passthrough = None
    noise_kl = kl(ref_lp, noise_lp).cpu().numpy()
    print(f"[reuse] {model_name} {genre} ctx={ctx} grid={grid_name}: prompt={len(ids)} ref {t_ref:.0f}s, "
          f"noise KL mean={noise_kl.mean():.2e} p95={np.quantile(noise_kl, 0.95):.2e}", flush=True)

    summary = {"model": model_name, "genre": genre, "ctx": ctx, "steps": steps, "prompt_tokens": len(ids),
               "ref_text": tok.decode(ref_tokens), "noise_kl_mean": float(noise_kl.mean()),
               "noise_kl_p95": float(np.quantile(noise_kl, 0.95)),
               "noise_top1": float(np.mean(np.array(noise_tok) == np.array(ref_tokens))), "conditions": {}}
    step_rows = [{"condition": "stock_sdpa_bf16", "step": i, "kl": float(noise_kl[i]),
                  "top1_agree": bool(noise_tok[i] == ref_tokens[i])} for i in range(steps)]
    head_frames = []

    for cfg in configs:
        name = cfg.name()
        t0 = time.perf_counter()
        lp, toks, recs = decode(model, cache, prefix_len, first, steps, eos_ids, ref_tokens, cfg)
        k = kl(ref_lp, lp).cpu().numpy()
        match = np.array(toks) == np.array(ref_tokens)
        R = {key: torch.stack([r[key] for r in recs]).cpu().numpy() for key in REC_KEYS}   # [S, L, Hkv]
        kept = R["kept"]
        k_loaded = np.ones_like(kept) if cfg.mode in ("fresh",) + reuse.VSKIP_MODES else kept  # K read in full
        sparse_layers = [l for l in range(L) if l not in cfg.dense_layers]
        refresh_rate = float(R["refresh"][1:, sparse_layers].mean())        # step 0 always refreshes
        row = {"condition": name, "mode": cfg.mode, "N": cfg.N, "theta": cfg.theta, "delta": cfg.delta,
               "max_age": cfg.max_age, "refresh": cfg.refresh,
               "kl_mean": float(k.mean()), "kl_median": float(np.median(k)),
               "kl_p95": float(np.quantile(k, 0.95)), "kl_max": float(k.max()),
               "kl_x_noise": float(k.mean() / noise_kl.mean()),
               "kl_p95_x_noise": float(np.quantile(k, 0.95) / np.quantile(noise_kl, 0.95)),
               "top1_agree": float(match.mean()),
               "first_div_tf": int(np.argmin(match)) if not match.all() else None,
               "k_skip": float(1 - k_loaded.mean()), "v_skip": float(1 - kept.mean()),
               "skip_nonrefresh": float(1 - kept[1:, sparse_layers][R["refresh"][1:, sparse_layers] == 0].mean())
               if (R["refresh"][1:, sparse_layers] == 0).any() else None,
               "refresh_rate": refresh_rate,
               "covered_min_mean": float(R["covered_min"][:, sparse_layers].mean()),
               "covered_min_p01": float(np.quantile(R["covered_min"][:, sparse_layers], 0.01)),
               "worst_step": int(np.argmax(k)), "seconds": round(time.perf_counter() - t0, 1)}
        summary["conditions"][name] = row
        for i in range(steps):
            step_rows.append({"condition": name, "step": i, "kl": float(k[i]), "top1_agree": bool(match[i]),
                              "k_loaded": float(k_loaded[i].mean()), "v_loaded": float(kept[i].mean()),
                              "refresh_rate": float(R["refresh"][i, sparse_layers].mean()),
                              "covered_min": float(R["covered_min"][i, sparse_layers].min()),
                              "covered_min_mean": float(R["covered_min"][i, sparse_layers].mean())})
        S, Lr, H = kept.shape
        head_frames.append(pd.DataFrame({
            "condition": name,
            "step": np.repeat(np.arange(S), Lr * H).astype(np.int16),
            "layer": np.tile(np.repeat(np.arange(Lr), H), S).astype(np.int16),
            "kv_head": np.tile(np.arange(H), S * Lr).astype(np.int16),
            **{key: R[key].reshape(-1).astype(np.float32) for key in REC_KEYS}}))
        print(f"  {name:24s} KL={row['kl_mean']:.2e} (x{row['kl_x_noise']:.2f}) p95={row['kl_p95']:.2e} "
              f"top1={row['top1_agree']:.3f} K-skip={row['k_skip']:.3f} V-skip={row['v_skip']:.3f} "
              f"refresh={refresh_rate:.3f} cov_min p01={row['covered_min_p01']:.3f} {row['seconds']}s", flush=True)

    sfx = "" if grid_name == "main" else f".{grid_name}"
    meta = {"model": model_name, "genre": genre, "ctx": ctx}
    pd.DataFrame(step_rows).assign(**meta).to_parquet(f"{stem}{sfx}.steps.parquet")
    pd.concat(head_frames).assign(**meta).to_parquet(f"{stem}{sfx}.heads.parquet")
    Path(f"{stem}{sfx}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    del cache
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=list(common.MODELS))
    ap.add_argument("--genre", nargs="+", default=list(common.GENRES))
    ap.add_argument("--ctx", nargs="+", type=int, default=[16384, 32768])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--grid", default="main", help="output suffix; 'main' and 'diag' have default conditions")
    ap.add_argument("--cond", nargs="*", default=[], help="conditions, see parse_cond (overrides the grid's)")
    ap.add_argument("--doc", nargs="*", default=[], help="extra documents: data/docs/a4/<name>.json (named by <name>)")
    args = ap.parse_args()
    tok, model = common.load(args.model)
    configs = grid(args.grid, args.cond)
    for ctx in args.ctx:
        if args.doc:
            for name in args.doc:
                doc = json.loads((common.ROOT / "data" / "docs" / "a4" / f"{name}.json").read_text())
                run(args.model, name, ctx, args.steps, tok, model, configs, args.grid,
                    with_ref_stats=False, doc=doc)
            continue
        for genre in args.genre:
            run(args.model, genre, ctx, args.steps, tok, model, configs, args.grid,
                with_ref_stats=(args.grid == "main"))


if __name__ == "__main__":
    main()
