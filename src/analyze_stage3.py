"""Stage 3 analysis: the gate (phase 1: decode step time at bs x ctx, step decomposition,
upper bound vs actual), the lab-vs-real-model check, and the phase-2 probe results (token
index for the always-exact heads, trigger before the pass)
-> results/tables/stage3_*.{csv,md}, results/figures/stage3_*.png. Robust to partial results.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_stage2 import auroc, save_fig, save_table

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LAB = RES / "lab"
E2E = RES / "e2e_s3"

# Llama-3.1-8B, bf16
LAYERS, HKV, HEAD_DIM, HIDDEN, FFN, VOCAB = 32, 8, 128, 4096, 14336, 128256
LAYER_W = 2 * (HIDDEN * (HIDDEN + 2 * HKV * HEAD_DIM + HIDDEN) + 3 * HIDDEN * FFN)      # bytes per layer
WEIGHT_BYTES = LAYERS * LAYER_W + 2 * HIDDEN * VOCAB                                  # + lm_head (embedding: B rows)
KV_PER_TOKEN = LAYERS * HKV * HEAD_DIM * 2 * 2
CEILING = 956e9                                                                       # streaming read, stage 2
MODES = ("sdpa", "dense", "vskip1", "refresh", "refresh_pre", "refresh_fused", "refresh_fused_ix", "replay")
LABEL = {"sdpa": "stock SDPA", "dense": "kvskip dense", "vskip1": "V-skip, 1 pass",
         "refresh": "refresh, stage 2 (Python planner, trigger after the pass)",
         "refresh_pre": "refresh, trigger before the pass (Python planner)",
         "refresh_fused": "refresh, planner in CUDA kernels",
         "refresh_fused_ix": "+ 5 induction heads on stored set + token index",
         "replay": "the same block lists, attention kernel only"}
# lab rounds: s3 = phase 1 (Python planners), s3f = first device planner, s3g = final device planner
# (+ induction heads on the token-pair index), s3h = final device planner (+ induction heads on the
# token index); a mode is taken from the last round that ran it, its gain against that round's dense
ROUNDS = ("s3", "s3f", "s3g", "s3h")


def load_rounds() -> dict:
    out = {}
    for tag in ROUNDS:
        for f in sorted(LAB.glob(f"{tag}_*.json")):
            r = json.loads(f.read_text())
            out.setdefault(tag, {})[r["ctx"]] = r
    return out


def load_lab() -> dict:
    """ctx -> merged result: each mode from the last round that ran it (dense and the KL / byte
    entries travel with it), so gains are always against the same round's dense."""
    rounds = load_rounds()
    merged = {}
    for tag in ROUNDS:
        for ctx, r in rounds.get(tag, {}).items():
            m = merged.setdefault(ctx, {"ctx": ctx, "batches": r["batches"], "docs": r["docs"], "head_ms": r["head_ms"],
                                        "step_ms": {}, "attn_ms": {}, "kl": {}, "bytes": {}, "round": {},
                                        "dense_ms": {}})
            for mode in r["step_ms"]:
                if mode in ("dense", "null"):
                    continue
                if mode == "replay" and "refresh_pre" not in r["step_ms"]:
                    continue                       # replay replays refresh_pre's lists; without it, nothing ran
                m["step_ms"][mode] = r["step_ms"][mode]
                m["attn_ms"][mode] = r["attn_ms"].get(mode, {})
                m["round"][mode] = tag
                m["dense_ms"][mode] = r["step_ms"]["dense"]
                src = mode if mode != "replay" else "refresh_pre"
                if src in r["kl"]:
                    m["kl"][mode] = r["kl"][src]
                if src in r["bytes"]:
                    m["bytes"][mode] = r["bytes"][src]
            m["step_ms"]["dense"] = r["step_ms"]["dense"]
            m["attn_ms"]["dense"] = r["attn_ms"]["dense"]
            m["dense_ms"]["dense"] = r["step_ms"]["dense"]
            m["round"]["dense"] = tag
            for key in ("dense", "sdpa"):
                if key in r["kl"]:
                    m["kl"][key] = r["kl"][key]
    return merged


def kv_read_fraction(r: dict, mode: str) -> float:
    """Fraction of the step's KV bytes the mode reads (layer 0 and the dense heads included)."""
    b = r["bytes"].get(mode, {})
    sparse = (LAYERS - 1) / LAYERS
    if "kv_read_frac_sparse_layers" in b:
        return 1 - sparse * (1 - b["kv_read_frac_sparse_layers"])
    if "v_read_frac_sparse_layers" in b:
        return 1 - sparse * (1 - b["v_read_frac_sparse_layers"]) / 2
    return 1.0


def gate(lab: dict):
    rows = []
    for ctx, r in sorted(lab.items()):
        for B in r["batches"]:
            Bs = str(B)
            kv = B * ctx * KV_PER_TOKEN
            for m in MODES:
                if m not in r["step_ms"]:
                    continue
                t = r["step_ms"][m][Bs]
                dense = r["dense_ms"][m][Bs]
                f = kv_read_fraction(r, m)
                attn = r["attn_ms"].get(m, {}).get(Bs, np.nan)
                rows.append({
                    "ctx": ctx, "bs": B, "mode": m, "round": r["round"][m], "ms_per_step": t,
                    "dense_ms_same_round": dense, "tokens_per_s": 1000 * B / t,
                    "attn_share": attn / t,
                    "kv_read": f,
                    "gain_vs_dense": 1 - t / dense,
                    "bound_bytes": (1 - f) * kv / (WEIGHT_BYTES + kv),
                    "bound_time": (1 - f) * r["attn_ms"]["dense"][Bs] / r["step_ms"]["dense"][Bs],
                    "kl_x_noise": r["kl"].get(m, {}).get("x_noise", np.nan),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    save_table(df.set_index(["ctx", "bs", "mode"]), "stage3_gate", floatfmt=".3f")

    # decomposition: bytes per step and where the time goes (dense kernel)
    dec = []
    for ctx, r in sorted(lab.items()):
        for B in r["batches"]:
            Bs = str(B)
            kv = B * ctx * KV_PER_TOKEN
            step = r["step_ms"]["dense"][Bs]
            attn = r["attn_ms"]["dense"][Bs]
            dec.append({"ctx": ctx, "bs": B, "weights_GB": WEIGHT_BYTES / 1e9, "kv_GB": kv / 1e9,
                        "kv_share_bytes": kv / (WEIGHT_BYTES + kv),
                        "ideal_ms": (WEIGHT_BYTES + kv) / CEILING * 1e3, "dense_ms": step,
                        "attn_ms": attn, "rest_ms": step - attn, "attn_share_time": attn / step,
                        "head_ms": r["head_ms"][Bs]})
    save_table(pd.DataFrame(dec).set_index(["ctx", "bs"]), "stage3_decomposition", floatfmt=".3g")

    # figure: the gate configurations (bs >= 8 at 32k / 64k), gain vs dense per mode + the byte bound
    cfgs = [(c, b) for c in sorted(lab) if c >= 32768 for b in lab[c]["batches"] if b >= 8]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(cfgs))
    shown = [m for m in ("vskip1", "refresh", "refresh_pre", "refresh_fused", "refresh_fused_ix", "replay")
             if m in df["mode"].unique()]
    w = 0.85 / (len(shown) + 1)
    lo = -60
    for i, m in enumerate(shown):
        vals = np.array([df[(df.ctx == c) & (df.bs == b) & (df["mode"] == m)]["gain_vs_dense"].mean()
                         for c, b in cfgs]) * 100
        ax.bar(x + i * w, np.maximum(vals, lo), w, label=LABEL[m])
        for xi, v in zip(x + i * w, vals):
            ax.text(xi, max(v, lo) + (1 if v >= 0 else -1), f"{v:+.0f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    bmode = "refresh_fused" if "refresh_fused" in df["mode"].unique() else "refresh_pre"
    bound = np.array([df[(df.ctx == c) & (df.bs == b) & (df["mode"] == bmode)]["bound_bytes"].mean()
                      for c, b in cfgs]) * 100
    ax.bar(x + len(shown) * w, bound, w, color="none", edgecolor="k", hatch="//",
           label="byte bound of the refresh policy's KV skip")
    for xi, v in zip(x + len(shown) * w, bound):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(10, color="grey", ls=":", lw=0.8)
    ax.axhline(20, color="grey", ls="--", lw=0.8)
    ax.set_ylim(lo - 8, 52)
    ax.set_xticks(x + 0.425 - w / 2, [f"{c // 1024}k, bs={b}" for c, b in cfgs])
    ax.set_ylabel("decode step time saved vs kvskip dense, %")
    ax.set_title("Stage 3 gate: Llama-3.1-8B decode step (layer-major lab, 8 documents; bars below -60% clipped)")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    save_fig(fig, "stage3_gate")
    return df


def lab_vs_real(lab: dict):
    rows = []
    for f in sorted(E2E.glob("*.json")):
        r = json.loads(f.read_text())
        ctx, bs = r["ctx"], r["batch"]
        if ctx not in lab or str(bs) not in lab[ctx]["step_ms"].get("dense", {}):
            continue
        for m, row in r["modes"].items():
            if m in lab[ctx]["step_ms"]:
                rows.append({"ctx": ctx, "bs": bs, "mode": m, "real_ms": row["ms_mean"],
                             "lab_ms": lab[ctx]["step_ms"][m][str(bs)],
                             "lab_over_real": lab[ctx]["step_ms"][m][str(bs)] / row["ms_mean"],
                             "real_kl_x_noise": row.get("kl_x_noise", np.nan)})
    if rows:
        save_table(pd.DataFrame(rows).set_index(["ctx", "bs", "mode"]), "stage3_lab_vs_real", floatfmt=".3g")


def kl_table(lab: dict):
    rows = []
    for ctx, r in sorted(lab.items()):
        for m, k in r["kl"].items():
            row = {"ctx": ctx, "mode": m, "kl_x_noise": k["x_noise"], "top1": k["top1"],
                   "kv_read": kv_read_fraction(r, m)}
            for d, v in zip(r["docs"], k["x_noise_by_doc"]):
                row[d] = v
            rows.append(row)
    if rows:
        save_table(pd.DataFrame(rows).set_index(["ctx", "mode"]), "stage3_lab_kl", floatfmt=".3g")


# ------------------------------------------------------------------- phase 2 (probe)

DENSE_HEADS = ((2, 5), (5, 2), (16, 5), (27, 1), (26, 3), (25, 1), (31, 2), (6, 7))
REUSE = RES / "reuse"


def load_heads(grid: str, cond_prefix: str) -> pd.DataFrame:
    frames = []
    for f in sorted(REUSE.glob(f"8b_*.{grid}.heads.parquet")):
        df = pd.read_parquet(f)
        frames.append(df[df["condition"].str.startswith(cond_prefix)])
    return pd.concat(frames) if frames else pd.DataFrame()


def index_heads(grid: str = "s3a"):
    """What the always-exact heads attend to, and how much of it a token-id index finds."""
    h = load_heads(grid, "estd")
    if h.empty or "idx_cov" not in h:
        return
    h = h[h.layer > 0]
    keys = ["idx_cov", "idx_tok", "idx_kept", "m_same", "m_after", "m_special", "m_newline", "m_sink", "m_local"]
    rows = []
    for (l, kv) in DENSE_HEADS:
        d = h[(h.layer == l) & (h.kv_head == kv)]
        row = {"head": f"{l}.{kv}", "docs": d["genre"].nunique() * d["ctx"].nunique()}
        row.update({k: d[k].mean() for k in keys})
        row["idx_cov_p05"] = d["idx_cov"].quantile(0.05)
        row["steps_idx_cov_ge_0.95"] = (d["idx_cov"] >= 0.95).mean()
        rows.append(row)
    allh = h.groupby(["layer", "kv_head"])[keys].mean()
    rows.append({"head": "all sparse heads (mean)", **allh.mean().to_dict(),
                 "idx_cov_p05": h["idx_cov"].quantile(0.05), "steps_idx_cov_ge_0.95": (h["idx_cov"] >= 0.95).mean()})
    save_table(pd.DataFrame(rows).set_index("head"), "stage3_index_heads", floatfmt=".3f")
    top = allh.sort_values("idx_tok", ascending=False).head(24)
    save_table(top, "stage3_index_top_heads", floatfmt=".3f")
    # map: token-level mass on "same token / token after it" positions, layer x KV head
    grid_ = allh["idx_tok"].unstack()
    fig, ax = plt.subplots(figsize=(5.5, 7))
    im = ax.imshow(grid_.values, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xlabel("KV head")
    ax.set_ylabel("layer")
    ax.set_yticks(range(len(grid_.index)), grid_.index, fontsize=6)
    for (l, kv) in DENSE_HEADS:
        ax.add_patch(plt.Rectangle((kv - 0.5, list(grid_.index).index(l) - 0.5), 1, 1, fill=False, ec="r", lw=1.5))
    ax.set_title("attention on earlier copies of the current token\nand the tokens after them (mean over q heads)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    save_fig(fig, "stage3_index_map")


def trigger_tables(grids=("s3a", "s3b")):
    rows = []
    for grid in grids:
        for f in sorted(REUSE.glob(f"8b_*.{grid}.summary.json")):
            r = json.loads(f.read_text())
            for name, c in r["conditions"].items():
                rows.append({"grid": grid, "doc": r["genre"], "ctx": r["ctx"], "condition": name,
                             "kl_x_noise": c["kl_x_noise"], "kl_p95_x_noise": c["kl_p95_x_noise"],
                             "top1": c["top1_agree"], "k_skip": c["k_skip"], "v_skip": c["v_skip"],
                             "refresh_rate": c["refresh_rate"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    save_table(df.set_index(["grid", "condition", "ctx", "doc"]).sort_index(), "stage3_trigger_by_doc", floatfmt=".3g")
    agg = df.groupby(["grid", "condition", "ctx"]).agg(
        kl_x_noise=("kl_x_noise", "mean"), kl_worst=("kl_x_noise", "max"), top1=("top1", "mean"),
        k_skip=("k_skip", "mean"), v_skip=("v_skip", "mean"), refresh_rate=("refresh_rate", "mean"), docs=("doc", "count"))
    save_table(agg, "stage3_trigger", floatfmt=".3g")
    # pre-pass vs post-pass estimate as predictors of a coverage loss (stale coverage < 0.95)
    h = load_heads("s3a", "estd")
    if not h.empty and "miss_p" in h:
        h = h[(h.layer > 0) & (h.step > 0)]
        lab = h["covered_stale_min"] < 0.95
        au = {"miss_d (after the pass)": auroc(h["miss_d"], lab), "miss_p (before the pass)": auroc(h["miss_p"], lab)}
        save_table(pd.DataFrame([au]), "stage3_trigger_auroc", floatfmt=".3f")


# ------------------------------------------------------------------- phase 4 (Qwen3-8B vs Llama-3.1-8B)

MODEL_NAMES = {"8b": "Llama-3.1-8B", "qwen3-8b": "Qwen3-8B"}


def layer_map(models=("8b", "qwen3-8b"), ctxs=(16384, 32768)):
    """Stage-1 statistics per layer for both models: sink mass, entropy, blocks for 0.9 / 0.99."""
    rows = []
    for model in models:
        for ctx in ctxs:
            hs = sorted((RES / "collect").glob(f"{model}_*_{ctx}.heads.parquet"))
            ks = sorted((RES / "collect").glob(f"{model}_*_{ctx}.kv.parquet"))
            if not hs:
                continue
            h = pd.concat([pd.read_parquet(f) for f in hs])
            k = pd.concat([pd.read_parquet(f) for f in ks])
            L = h.layer.max() + 1
            gh, gk = h.groupby("layer"), k.groupby("layer")
            t = pd.DataFrame({"sink_mass_med": gh["sink_mass"].median(), "entropy_norm_med": gh["entropy_norm"].median(),
                              "blocks_to_0.9_med": gh["blocks_to_0.9"].median(),
                              "head_frac_to_0.99": gh["frac_blocks_to_0.99"].mean(),
                              "union_frac_to_0.99": gk["union_frac_blocks_to_0.99"].mean()})
            t["model"], t["ctx"], t["depth"] = model, ctx, t.index / (L - 1)
            rows.append(t.reset_index())
    if not rows:
        return None
    df = pd.concat(rows)
    save_table(df.set_index(["model", "ctx", "layer"]), "stage3_layer_map", floatfmt=".3g")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for model, ls in zip(models, ("-", "--")):
        d = df[(df.model == model) & (df.ctx == max(ctxs))]
        if d.empty:
            d = df[df.model == model]
        if d.empty:
            continue
        for ax, col, title in zip(axes, ("union_frac_to_0.99", "sink_mass_med", "blocks_to_0.9_med"),
                                  ("KV-head blocks for 0.99 of the mass (GQA union)", "sink mass (first 4 tokens)",
                                   "blocks for 0.9 of the mass (median head)")):
            ax.plot(d.depth, d[col], ls, marker=".", label=f"{MODEL_NAMES[model]} ({d.ctx.iloc[0] // 1024}k)")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("layer / (layers - 1)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "stage3_layer_map")
    return df


def jumps(models=("8b", "qwen3-8b")):
    """Under the timer-32 block set: in how many KV heads the set newly stops covering 95% of the
    mass in the same step (a jump), against the median step."""
    rows, per_layer = [], []
    for model in models:
        for f in sorted(REUSE.glob(f"{model}_*.diag.heads.parquet")):
            h = pd.read_parquet(f)
            g = h[(h.condition == "timer_N32") & (h.layer > 0)]
            if g.empty:
                continue
            lost = g.assign(lost=g.covered_stale_min < 0.95).pivot_table(
                index="step", columns=["layer", "kv_head"], values="lost").astype(bool)
            steps = [t for t in range(2, len(lost)) if lost.index[t] % 32 >= 2]   # both steps stale
            newly = np.array([int((lost.iloc[t] & ~lost.iloc[t - 1]).sum()) for t in steps])
            doc, ctx = f.name.split("_")[1], int(f.name.split("_")[2].split(".")[0])
            rows.append({"model": model, "doc": doc, "ctx": ctx, "sparse_kv_heads": lost.shape[1],
                         "max_newly_lost_one_step": int(newly.max()), "median_newly_lost": float(np.median(newly)),
                         "p95_newly_lost": float(np.quantile(newly, 0.95)),
                         "max_share": float(newly.max() / lost.shape[1])})
            loss = g[g.refresh == 0].assign(lost=lambda d: d.covered_stale_min < 0.95)
            pl = loss.groupby("layer")["lost"].mean().rename("p_stale_cov_lt_0.95").reset_index()
            pl["model"], pl["doc"], pl["ctx"] = model, doc, ctx
            per_layer.append(pl)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    save_table(df.set_index(["model", "ctx", "doc"]), "stage3_jumps", floatfmt=".3g")
    pl = pd.concat(per_layer)
    t = pl.pivot_table(index="layer", columns=["model", "ctx"], values="p_stale_cov_lt_0.95", aggfunc="mean")
    save_table(t, "stage3_jump_layers", floatfmt=".3f")
    return df


def induction_map(model: str, grid: str = "diag"):
    """Heads by attention on 'the token after an earlier copy of the current token'."""
    files = sorted(REUSE.glob(f"{model}_*.{grid}.heads.parquet"))
    if not files:
        return None
    h = pd.concat([pd.read_parquet(f) for f in files])
    if "m_after" not in h:
        return None
    h = h[h.condition == h.condition.iloc[0]]
    per = h.groupby(["layer", "kv_head"])[["m_after", "m_same", "idx_cov", "idx2_cov", "m_sink", "m_local"]].mean()
    per = per.sort_values("m_after", ascending=False)
    save_table(per.head(16), f"stage3_induction_{model}", floatfmt=".3f")
    return per


def main():
    index_heads()
    trigger_tables()
    trigger_tables(grids=("s3a", "s3b"))
    layer_map()
    jumps()
    for model in ("8b", "qwen3-8b"):
        induction_map(model, "s3a" if model == "8b" else "diag")
    lab = load_lab()
    if lab:
        gate(lab)
        lab_vs_real(lab)
        kl_table(lab)
    for t in ("stage3_gate", "stage3_decomposition", "stage3_lab_vs_real", "stage3_lab_kl",
              "stage3_index_heads", "stage3_trigger", "stage3_trigger_auroc", "stage3_jumps",
              "stage3_induction_8b", "stage3_induction_qwen3-8b"):
        p = ROOT / "results" / "tables" / f"{t}.md"
        if p.exists():
            print(f"== {t}\n{p.read_text()}\n")


if __name__ == "__main__":
    main()
