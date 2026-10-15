"""Prefill a long document, then greedy-decode N tokens through the probe and
record attention statistics for every decode step x layer x head.

Outputs (results/collect/<model>_<genre>_<ctx>.*):
  heads.parquet  one row per (step, layer, head): metrics.head_stats per query head
  kv.parquet     one row per (step, layer, kv_head): GQA-union metrics
  hist.npz       per-layer 2D histogram of (pre-exp gap, log10 block weight)
  meta.json      prompt/decode info and the generated text

Input: data/docs/<genre>.json (src/prepare_data.py), models/<model>.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
import metrics
from probe_attn import STATE

OUT = common.ROOT / "results" / "collect"


def stack_layers(per_layer: dict, n_layers: int) -> dict:
    """{layer: {metric: [Hkv, G] or [Hkv]}} -> {metric: [L, H] or [L, Hkv]} (GPU)."""
    names = per_layer[0].keys()
    return {k: torch.stack([per_layer[l][k].flatten() for l in range(n_layers)]) for k in names}


def greedy_token(logits: torch.Tensor, eos_ids: list[int]) -> tuple[int, bool]:
    """Greedy pick with EOS suppressed, so every run has exactly N decode steps."""
    top = int(logits.argmax())
    if top not in eos_ids:
        return top, False
    masked = logits.clone()
    masked[eos_ids] = float("-inf")
    return int(masked.argmax()), True


def run(model_name: str, genre: str, ctx: int, steps: int, tok=None, model=None, out=OUT):
    if model is None:
        tok, model = common.load(model_name)
    eos_ids = common.eos_ids(model)
    L = model.config.num_hidden_layers
    ids, n_units = common.build_prompt(tok, common.load_doc(genre), ctx - steps)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache = common.prefill(model, ids[:-1])
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    STATE.reset()
    STATE.collect = True
    q_rows, kv_rows, generated, eos_steps = [], [], [], []
    token = ids[-1]
    t0 = time.perf_counter()
    for i in range(steps):
        logits = common.step(model, token, cache)
        q_stats, kv_stats, _ = STATE.take_step()
        q = {k: v.cpu().numpy() for k, v in stack_layers(q_stats, L).items()}
        kv = {k: v.cpu().numpy() for k, v in stack_layers(kv_stats, L).items()}
        q_rows.append(q)
        kv_rows.append(kv)
        token, suppressed = greedy_token(logits, eos_ids)
        if suppressed:
            eos_steps.append(i)
        generated.append(token)
    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t0
    STATE.collect = False

    T0 = len(ids)   # context length (incl. the fed token) at decode step 0
    heads = frames(q_rows, model_name, genre, ctx, T0, "head")
    kvs = frames(kv_rows, model_name, genre, ctx, T0, "kv_head")
    G = model.config.num_attention_heads // model.config.num_key_value_heads
    heads["kv_head"] = heads["head"] // G

    stem = out / f"{model_name}_{genre}_{ctx}"
    out.mkdir(parents=True, exist_ok=True)
    heads.to_parquet(f"{stem}.heads.parquet")
    kvs.to_parquet(f"{stem}.kv.parquet")
    hist = np.stack([STATE.hist[l].cpu().numpy().reshape(metrics.GAP_EDGES[2], metrics.LOGW_EDGES[2])
                     for l in range(L)])
    np.savez_compressed(f"{stem}.hist.npz", hist=hist, gap_edges=metrics.GAP_EDGES,
                        logw_edges=metrics.LOGW_EDGES)
    meta = {
        "model": model_name, "genre": genre, "ctx": ctx, "steps": steps,
        "prompt_tokens": len(ids), "doc_units": n_units,
        "T_first": T0, "T_last": T0 + steps - 1,
        "block": metrics.BLOCK, "sink_tokens": metrics.SINK_TOKENS,
        "eos_suppressed_at_steps": eos_steps,
        "prefill_s": round(t_prefill, 2), "decode_s": round(t_decode, 2),
        "question": tok.decode(ids[-60:]),
        "generated": tok.decode(generated),
        "generated_ids": generated,
    }
    (out / f"{model_name}_{genre}_{ctx}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[collect] {model_name} {genre} ctx={ctx}: prompt={len(ids)} prefill={t_prefill:.1f}s "
          f"decode={t_decode:.1f}s ({1000 * t_decode / steps:.0f} ms/step) eos_suppressed={eos_steps[:5]}", flush=True)
    print("   ", repr(meta["generated"][:300]))
    del cache
    torch.cuda.empty_cache()
    return meta


def frames(rows: list[dict], model_name, genre, ctx, T0, unit: str) -> pd.DataFrame:
    """rows[step][metric] = [L, n] -> long DataFrame (step, layer, <unit>, metrics...)."""
    S = len(rows)
    Lyr, n = next(iter(rows[0].values())).shape
    cols = {k: np.stack([r[k] for r in rows]).reshape(-1).astype(np.float32) for k in rows[0]}
    step = np.repeat(np.arange(S), Lyr * n)
    df = pd.DataFrame({
        "model": model_name, "genre": genre, "ctx": ctx,
        "step": step.astype(np.int16),
        "T": (T0 + step).astype(np.int32),
        "layer": np.tile(np.repeat(np.arange(Lyr), n), S).astype(np.int16),
        unit: np.tile(np.arange(n), S * Lyr).astype(np.int16),
        **cols,
    })
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(common.MODELS))
    ap.add_argument("--genre", nargs="+", default=list(common.GENRES))
    ap.add_argument("--ctx", nargs="+", type=int, required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    tok, model = common.load(args.model)
    for ctx in args.ctx:
        for genre in args.genre:
            run(args.model, genre, ctx, args.steps, tok, model, args.out)


if __name__ == "__main__":
    main()
