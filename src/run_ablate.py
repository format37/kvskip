"""First ablation (no custom kernel yet): drop KV blocks in the probe's decode
attention and measure the damage against the exact fp32 probe.

For every document: one prefill (unmasked, shared by all conditions), one
exact reference greedy run, then for each condition
  * teacher-forced run on the reference tokens -> KL(ref || masked) per step,
    first step where the masked argmax differs from the reference token
    (= first divergence of free-running greedy, up to fp nondeterminism),
  * free-running greedy run -> the actually generated text and its first
    divergence step.
Conditions: criterion {weight (oracle), preexp (QK-max)} x tau {1e-4, 1e-3, 1e-2}
x scope {head, union (GQA)}. Noise floor: stock HF bf16 sdpa/eager decode,
teacher-forced the same way.

Outputs results/ablate/<model>_<genre>_<ctx>.{steps.parquet,layers.parquet,summary.json}
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
import metrics
from probe_attn import STATE
from run_collect import greedy_token

OUT = common.ROOT / "results" / "ablate"
CRITERIA = ("weight", "preexp")
SCOPES = ("head", "union")
GRIDS = {
    # the kickoff grid: per-block thresholds
    "main": list(itertools.product(CRITERIA, metrics.TAUS, SCOPES)),
    # extra: oracle top-p -- keep the fewest blocks holding this share of the weight
    "mass": [("mass", level, scope) for level in (0.9, 0.99, 0.999) for scope in SCOPES],
}


def kl(ref_lp: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
    return (ref_lp.exp() * (ref_lp - lp)).sum(-1)


def decode(model, cache, prefix_len, first_token, steps, eos_ids, forced=None):
    """Greedy (forced=None) or teacher-forced decode from a cropped cache.

    Returns log-probs [steps, V], chosen tokens and per-step skip records.
    """
    extra = cache.get_seq_length() - prefix_len
    if extra > 0:
        cache.crop(-extra)              # negative = number of tokens to drop (transformers 5.x)
    lps, tokens, skips = [], [], []
    token = first_token
    for i in range(steps):
        logits = common.step(model, token, cache)
        lps.append(torch.log_softmax(logits, -1))
        _, _, skip = STATE.take_step()
        if skip:
            L = len(skip)
            frac = torch.stack([skip[l][0].flatten() for l in range(L)])      # [L, H]
            mass = torch.stack([skip[l][1].flatten() for l in range(L)])
            skips.append((frac.mean(1), mass.mean(1), mass.amax(1)))
        pick, _ = greedy_token(logits, eos_ids)
        tokens.append(pick)
        token = forced[i] if forced is not None else pick
    return torch.stack(lps), tokens, skips


def run(model_name, genre, ctx, steps, tok, model, grid="main"):
    eos_ids = list(model.generation_config.eos_token_id)
    ids, _ = common.build_prompt(tok, common.load_doc(genre), ctx - steps)
    prefix_len = len(ids) - 1
    cache = common.prefill(model, ids[:-1])
    first = ids[-1]

    STATE.reset()
    STATE.mask, STATE.passthrough, STATE.collect = None, None, False
    t0 = time.perf_counter()
    ref_lp, ref_tokens, _ = decode(model, cache, prefix_len, first, steps, eos_ids)
    t_ref = time.perf_counter() - t0
    forced = ref_tokens          # feed y_0..y_{N-2}; step i predicts y_i

    step_rows, layer_rows, summary = [], [], {
        "model": model_name, "genre": genre, "ctx": ctx, "steps": steps,
        "prompt_tokens": len(ids), "ref_text": tok.decode(ref_tokens), "conditions": {},
    }

    def record(name, cond, tf_lp, tf_tokens, skips, free_tokens=None):
        k = kl(ref_lp, tf_lp).cpu().numpy()
        match = np.array(tf_tokens) == np.array(ref_tokens)
        first_div_tf = int(np.argmin(match)) if not match.all() else None
        row = {"condition": name, **cond,
               "kl_mean": float(k.mean()), "kl_median": float(np.median(k)),
               "kl_p95": float(np.quantile(k, 0.95)), "kl_max": float(k.max()),
               "top1_agree": float(match.mean()), "first_div_tf": first_div_tf}
        if skips:
            frac = torch.stack([s[0] for s in skips]).cpu().numpy()            # [steps, L]
            mass = torch.stack([s[1] for s in skips]).cpu().numpy()
            mass_max = torch.stack([s[2] for s in skips]).cpu().numpy()
            row["frac_blocks_skipped"] = float(frac.mean())
            row["mass_dropped_mean"] = float(mass.mean())
            row["mass_dropped_max"] = float(mass_max.max())
            for l in range(frac.shape[1]):
                layer_rows.append({"condition": name, **cond, "layer": l,
                                   "frac_blocks_skipped": float(frac[:, l].mean()),
                                   "mass_dropped_mean": float(mass[:, l].mean()),
                                   "mass_dropped_max": float(mass_max[:, l].max())})
        if free_tokens is not None:
            fm = np.array(free_tokens) == np.array(ref_tokens)
            row["first_div_free"] = int(np.argmin(fm)) if not fm.all() else None
            row["free_text"] = tok.decode(free_tokens)
        summary["conditions"][name] = row
        for i in range(steps):
            step_rows.append({"condition": name, **cond, "step": i, "kl": float(k[i]),
                              "top1_agree": bool(match[i]),
                              "frac_blocks_skipped": float(frac[i].mean()) if skips else 0.0,
                              "mass_dropped_mean": float(mass[i].mean()) if skips else 0.0})
        print(f"  {name:22s} KL mean={row['kl_mean']:.2e} p95={row['kl_p95']:.2e} max={row['kl_max']:.2e} "
              f"agree={row['top1_agree']:.3f} div_tf={first_div_tf} div_free={row.get('first_div_free')} "
              f"skipped={row.get('frac_blocks_skipped', 0):.3f}", flush=True)

    print(f"[ablate] {model_name} {genre} ctx={ctx} grid={grid}: prompt={len(ids)}, ref run {t_ref:.1f}s", flush=True)
    for impl in (("sdpa", "eager") if grid == "main" else ()):
        STATE.passthrough = impl
        lp, toks, _ = decode(model, cache, prefix_len, first, steps, eos_ids, forced)
        _, free, _ = decode(model, cache, prefix_len, first, steps, eos_ids)
        record(f"stock_{impl}_bf16", {"criterion": f"stock_{impl}", "tau": None, "scope": None},
               lp, toks, [], free)
    STATE.passthrough = None

    for criterion, tau, scope in GRIDS[grid]:
        STATE.mask = (criterion, tau, scope)
        lp, toks, skips = decode(model, cache, prefix_len, first, steps, eos_ids, forced)
        _, free, _ = decode(model, cache, prefix_len, first, steps, eos_ids)
        record(f"{criterion}_{tau:g}_{scope}", {"criterion": criterion, "tau": tau, "scope": scope},
               lp, toks, skips, free)
    STATE.mask = None

    stem = OUT / (f"{model_name}_{genre}_{ctx}" + ("" if grid == "main" else f".{grid}"))
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"model": model_name, "genre": genre, "ctx": ctx}
    pd.DataFrame(step_rows).assign(**meta).to_parquet(f"{stem}.steps.parquet")
    pd.DataFrame(layer_rows).assign(**meta).to_parquet(f"{stem}.layers.parquet")
    Path(f"{stem}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    del cache
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(common.MODELS))
    ap.add_argument("--genre", nargs="+", default=list(common.GENRES))
    ap.add_argument("--ctx", nargs="+", type=int, required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--grid", choices=list(GRIDS), default="main",
                    help="main: kickoff threshold grid + stock-kernel noise floor; mass: extra top-p grid")
    args = ap.parse_args()
    tok, model = common.load(args.model)
    for ctx in args.ctx:
        for genre in args.genre:
            run(args.model, genre, ctx, args.steps, tok, model, args.grid)


if __name__ == "__main__":
    main()
