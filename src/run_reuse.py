"""Stage 2 / A1 + A2: block-selection reuse ablation (see src/reuse.py).

For every document: one prefill (shared), one exact fp32 reference greedy run
(which also accumulates the A2 locality statistics), the stock bf16 SDPA noise
floor, then each reuse condition teacher-forced on the reference tokens ->
KL(ref || cond) per step, top-1 agreement, bytes skipped and per
(step, layer, KV head) records of coverage / refresh / query drift.

Outputs results/reuse/<model>_<genre>_<ctx>[.<grid>].{steps.parquet,heads.parquet,summary.json}
and <model>_<genre>_<ctx>.locality.npz (reference pass only).

Input: data/docs/<genre>.json (src/prepare_data.py) or, with --doc, data/docs/{a4,s3}/<name>.json
(src/prepare_a4.py, src/prepare_s3.py); models/<model>.
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
            "miss_c", "miss_d", "calib", "auto_ix") + reuse.DIAG_KEYS


def _heads(val: str) -> tuple:
    return tuple(tuple(int(x) for x in lh.split(".")) for lh in val.split(","))


def parse_cond(spec: str) -> ReuseConfig:
    """mode[:value[:max_age]][:flag][:key=value...]. Positional: the timer period, oracle theta or
    trigger delta, then max age (quest / static / topk: the byte budget). Flags: masked (refresh
    applies the fresh set), free (calibration may go negative). Options:
      dh=L.H,...          always-exact KV heads           ix= / ixb= / ixu= / ixub= token-index heads
                          (b: token-pair index, u: index union -- the blocks join the stored set)
      cal=a               stage 4: online calibration of the pre-pass estimate, EMA rate a
      hold=N              ... applied from step N on      warm=W.A  max age A during the first W steps
      aix=x               ... heads whose correction exceeds x join the token index
      kind=logit|add      ... the correction is a log-odds offset (a ratio) or added to the miss share
      dl=0,1 | dl=none    dense layers (default 0)        p=0.95    top-p level of the stored set
      match=<condition>   quest / static / topk: budget = that condition's bytes in the same run
      sub=16              quest: sub-block size of the key min/max
      union=norm|raw|soft|rr  quest: how a KV head combines its query heads' bounds (ReuseConfig.quest_union)
      native=1            quest: select and read sub-block pages (Quest's granularity) instead of 64-blocks
    e.g. timer:16, oracle:0.98:32, estd:0.02:32:dh=2.5,25.3, estp:0.05:32:cal=0.3:warm=32.8,
    quest:0.4, quest:match=estp_delta0.05_age32_cal0.3."""
    parts = spec.split(":")
    mode, pos, extra = parts[0], [], {}
    for part in parts[1:]:
        if "=" not in part:
            if part == "masked":
                extra["refresh"] = "masked"
            elif part == "free":
                extra["calib_free"] = True
            else:
                pos.append(part)
            continue
        key, val = part.split("=", 1)
        if key == "dh":
            extra["dense_heads"] = _heads(val)
        elif key in ("ix", "ixb", "ixu", "ixub"):
            extra["index_heads"] = _heads(val)
            if key in ("ixb", "ixub"):
                extra["index_kind"] = "bigram"
            if key in ("ixu", "ixub"):
                extra["index_union"] = True
        elif key == "cal":
            extra["calib"] = float(val)
        elif key == "hold":
            extra["calib_hold"] = int(val)
        elif key == "warm":
            w, a = val.split(".")
            extra["calib_warm"] = (int(w), int(a))
        elif key == "aix":
            extra["calib_auto_ix"] = float(val)
        elif key == "kind":
            extra["calib_kind"] = val
        elif key == "dl":
            extra["dense_layers"] = () if val in ("", "none") else tuple(int(x) for x in val.split(","))
        elif key == "p":
            extra["level"] = float(val)
        elif key == "match":
            extra["match"] = val
        elif key == "sub":
            extra["quest_sub"] = int(val)
        elif key == "union":
            extra["quest_union"] = val
        elif key == "native":
            extra["quest_native"] = val in ("1", "true", "yes")
        else:
            raise ValueError(f"unknown option {key!r} in {spec!r}")
    if mode == "fresh":
        return ReuseConfig("fresh", **extra)
    if mode in reuse.BUDGET_MODES:
        if pos:
            extra["budget"] = float(pos[0])
        assert ("budget" in extra) != ("match" in extra), f"{spec}: give a budget or match=, not both"
        return ReuseConfig(mode, **extra)
    if len(pos) > 1:
        extra["max_age"] = int(pos[1])
    if mode == "timer":
        return ReuseConfig("timer", N=int(pos[0]), **extra)
    if mode == "oracle":
        return ReuseConfig("oracle", theta=float(pos[0]), **extra)
    return ReuseConfig(mode, delta=float(pos[0]), **extra)


def meta_frac(cfg: ReuseConfig, D: int, G: int) -> float:
    """Metadata a sparse layer reads or writes per step, as a share of its K+V bytes (bf16 K and
    V: 4 D bytes per token and KV head, 256 D per 64-block).
      refresh (est*): the planner kernels as implemented -- fp32 block key sums (4 D bytes per
        block), the fp32 references lse_ref / cs_ref and centroid scores (3 G floats read or
        written), the block stats written by the attention pass and read by `update` (2 x 2 G
        floats), keep + block list (5 bytes): 2.0% for D = 128, G = 4 (1.2% with bf16 sums);
      quest: per-channel key min and max per sub-block in bf16 (4 D bytes per sub-block): 1/sub;
      static / topk / fresh / V-skip / timer: none counted (topk is an oracle)."""
    if cfg.mode in ("estc", "estd", "estp", "drift", "oracle"):
        return (4 * D + 4 * 9 * G + 5) / (256 * D)
    if cfg.mode == "quest":
        return 1.0 / cfg.quest_sub
    return 0.0


def bytes_of(cfg: ReuseConfig, kept, k_loaded, L: int, D: int, G: int) -> dict:
    """K+V bytes of a decode run as a share of dense attention's, over all layers and steps."""
    n_sparse = L - len([l for l in cfg.dense_layers if l < L])
    blocks = float((k_loaded.mean() + kept.mean()) / 2)
    meta = meta_frac(cfg, D, G) * n_sparse / L
    return {"bytes_blocks": blocks, "bytes_meta": meta, "bytes_total": blocks + meta}


def set_budget(cfg: ReuseConfig, target: float, L: int, D: int, G: int, steps: int):
    """quest / static / topk: the share of blocks a sparse layer reads per step so that blocks +
    metadata over all layers and steps come to `target` of dense attention's bytes (static reads
    its first step in full, to select)."""
    dense = len([l for l in cfg.dense_layers if l < L])
    f = (L * target - dense) / (L - dense) - meta_frac(cfg, D, G)
    if cfg.mode == "static":
        f = (steps * f - 1.0) / (steps - 1)
    cfg.block_frac = max(0.0, f)
    cfg.budget_target = target


GRIDS = {
    "main": ["fresh", "timer:4", "timer:8", "timer:16", "timer:32", "oracle:0.98", "oracle:0.95"],
    "diag": ["timer:32", "vtau:0.0001", "vsink:0.0001", "vprev:0.0001", "vprev2:0.0001"],
}


def grid(name: str, conds=()) -> list[ReuseConfig]:
    return [parse_cond(c) for c in (conds or GRIDS[name])]


def vocab_classes(tok, device, vocab_size: int) -> dict:
    """Token classes for the stage-3 diagnostics: special tokens and tokens containing a newline
    (sized to the embedding table, which can be larger than the tokenizer's vocabulary)."""
    n = len(tok)
    special = torch.zeros(max(n, vocab_size), dtype=torch.bool)
    special[sorted(tok.all_special_ids + [i for i in range(n) if tok.convert_ids_to_tokens(i).startswith("<|")])] = True
    newline = torch.zeros_like(special)
    newline[:n] = torch.tensor(["\n" in tok.decode([i]) for i in range(n)])
    return {"special": special.to(device), "newline": newline.to(device)}


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
        IDS[prefix_len + i] = token
        STATE.ids = IDS[:prefix_len + i + 1]
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
    global IDS
    eos_ids = common.eos_ids(model)
    ids, _ = common.build_prompt(tok, doc if doc is not None else common.load_doc(genre), ctx - steps)
    prefix_len = len(ids) - 1
    IDS = torch.zeros(prefix_len + steps, dtype=torch.long, device=model.device)
    IDS[:prefix_len] = torch.tensor(ids[:-1], device=model.device)
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
    D = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    G = model.config.num_attention_heads // model.config.num_key_value_heads
    spent = {}                                   # condition name -> bytes (for match=)

    for cfg in configs:
        if cfg.mode in reuse.BUDGET_MODES:
            target = cfg.budget if cfg.match is None else spent[cfg.match]["bytes_total"]
            set_budget(cfg, target, L, D, G, steps)
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
        spent[name] = bytes_of(cfg, kept, k_loaded, L, D, G)
        row = {"condition": name, "mode": cfg.mode, "N": cfg.N, "theta": cfg.theta, "delta": cfg.delta,
               "max_age": cfg.max_age, "refresh": cfg.refresh, **spent[name],
               "block_frac": cfg.block_frac, "budget_target": cfg.budget_target, "match": cfg.match,
               "calib_final_mean": float(R["calib"][-1, sparse_layers].mean()),
               "calib_heads_gt1": int((R["calib"][-1, sparse_layers] > 1.0).sum()),
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
              f"bytes={row['bytes_total']:.3f} refresh={refresh_rate:.3f} "
              f"cov_min p01={row['covered_min_p01']:.3f} {row['seconds']}s", flush=True)

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
    ap.add_argument("--doc", nargs="*", default=[],
                    help="other documents by name: data/docs/{a4,s3}/<name>.json (instead of --genre)")
    args = ap.parse_args()
    tok, model = common.load(args.model)
    STATE.vocab = vocab_classes(tok, model.device, model.config.vocab_size)
    configs = grid(args.grid, args.cond)
    for ctx in args.ctx:
        if args.doc:
            for name in args.doc:
                run(args.model, name, ctx, args.steps, tok, model, configs, args.grid,
                    with_ref_stats=False, doc=common.load_named_doc(name))
            continue
        for genre in args.genre:
            run(args.model, genre, ctx, args.steps, tok, model, configs, args.grid,
                with_ref_stats=(args.grid == "main"))


if __name__ == "__main__":
    main()
