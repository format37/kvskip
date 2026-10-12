"""Turn results/collect and results/ablate into the digest's tables and figures.

Writes results/figures/*.png, results/tables/*.{csv,md} and prints the key
numbers. Robust to partial results: whatever runs exist are analysed.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
GENRES = ("prose", "code", "dialog")
TAUS = ("0.0001", "0.001", "0.01")


def p95(x):
    return np.quantile(x, 0.95)


def load_collect(src: Path):
    heads = pd.concat([pd.read_parquet(f) for f in sorted(src.glob("*.heads.parquet"))], ignore_index=True)
    kv = pd.concat([pd.read_parquet(f) for f in sorted(src.glob("*.kv.parquet"))], ignore_index=True)
    hist = {}
    for f in sorted(src.glob("*.hist.npz")):
        model, genre, ctx = f.name.split(".")[0].split("_")
        z = np.load(f)
        hist[(model, genre, int(ctx))] = z["hist"]
        edges = (tuple(z["gap_edges"]), tuple(z["logw_edges"]))
    meta = {}
    for f in sorted(src.glob("*.meta.json")):
        m = json.loads(f.read_text())
        meta[(m["model"], m["genre"], m["ctx"])] = m
    return heads, kv, hist, edges, meta


def save_table(df: pd.DataFrame, name: str, floatfmt=".3g"):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv")
    (TAB / f"{name}.md").write_text(df.to_markdown(floatfmt=floatfmt))
    return df


# ---------------------------------------------------------------- overview

def overview(heads, kv):
    """Pooled over layers, heads and decode steps: per model x ctx x genre."""
    g = heads.groupby(["model", "ctx", "genre"])
    t = pd.DataFrame({
        "T_mean": g["T"].mean(),
        "nblocks": g["T"].mean() / 64,
        "entropy_norm_med": g["entropy_norm"].median(),
        "sink_mass_med": g["sink_mass"].median(),
        "blocks_to_0.9_med": g["blocks_to_0.9"].median(),
        "blocks_to_0.99_med": g["blocks_to_0.99"].median(),
        "blocks_to_0.99_p95": g["blocks_to_0.99"].quantile(0.95),
        "frac_to_0.99_mean": g["frac_blocks_to_0.99"].mean(),
        "frac_to_0.99_med": g["frac_blocks_to_0.99"].median(),
        "frac_lt_1e-3_mean": g["frac_blocks_lt_0.001"].mean(),
        "frac_lt_1e-2_mean": g["frac_blocks_lt_0.01"].mean(),
        "mass_lost_lt_1e-3_mean": g["mass_lost_lt_0.001"].mean(),
        "mass_lost_lt_1e-2_mean": g["mass_lost_lt_0.01"].mean(),
    })
    gk = kv.groupby(["model", "ctx", "genre"])
    t["union_frac_to_0.99_mean"] = gk["union_frac_blocks_to_0.99"].mean()
    t["union_frac_lt_1e-3_mean"] = gk["union_frac_blocks_lt_0.001"].mean()
    t["union_frac_lt_1e-2_mean"] = gk["union_frac_blocks_lt_0.01"].mean()
    return save_table(t, "overview_by_run")


def scaling(heads):
    """Context-length scaling, pooled over genres."""
    g = heads.groupby(["model", "ctx"])
    t = pd.DataFrame({
        "nblocks": (g["T"].mean() / 64).round(0),
        "blocks_to_0.9_med": g["blocks_to_0.9"].median(),
        "blocks_to_0.9_p95": g["blocks_to_0.9"].quantile(0.95),
        "blocks_to_0.99_med": g["blocks_to_0.99"].median(),
        "blocks_to_0.99_p95": g["blocks_to_0.99"].quantile(0.95),
        "frac_to_0.99_med": g["frac_blocks_to_0.99"].median(),
        "frac_to_0.99_mean": g["frac_blocks_to_0.99"].mean(),
        "entropy_norm_med": g["entropy_norm"].median(),
        "sink_mass_med": g["sink_mass"].median(),
    })
    return save_table(t, "scaling_by_ctx")


# ---------------------------------------------------- layer x head (8B 32k)

def layer_head(heads, model="8b", ctx=32768):
    d = heads[(heads.model == model) & (heads.ctx == ctx)]
    if d.empty:
        return None
    per_genre = d.groupby(["genre", "layer", "head"]).agg(
        entropy_norm=("entropy_norm", "median"),
        sink_mass=("sink_mass", "median"),
        blocks_to_099=("blocks_to_0.99", "median"),
        blocks_to_099_p95=("blocks_to_0.99", p95),
        frac_to_099=("frac_blocks_to_0.99", "median"),
        local_mass=("local_mass", "median"),
        spearman=("spearman_blockmax_vs_weight", "median"),
    )
    wide = per_genre.unstack("genre")
    wide.columns = [f"{m}_{g}" for m, g in wide.columns]
    pooled = per_genre.groupby(["layer", "head"]).mean()
    table = pooled.join(wide)
    table.insert(0, "kv_head", table.index.get_level_values("head") // 4)
    save_table(table, f"{model}_{ctx}_layer_head")

    by_layer = d.groupby("layer").agg(
        entropy_norm_med=("entropy_norm", "median"),
        entropy_norm_p95=("entropy_norm", p95),
        sink_mass_med=("sink_mass", "median"),
        sink_mass_p95=("sink_mass", p95),
        local_mass_med=("local_mass", "median"),
        blocks_to_09_med=("blocks_to_0.9", "median"),
        blocks_to_099_med=("blocks_to_0.99", "median"),
        blocks_to_099_mean=("blocks_to_0.99", "mean"),
        blocks_to_099_p95=("blocks_to_0.99", p95),
    )
    save_table(by_layer, f"{model}_{ctx}_by_layer")

    fig, axes = plt.subplots(3, 4, figsize=(19, 13), constrained_layout=True)
    specs = [("entropy_norm", "entropy / log2 T", "viridis", (0, 1)),
             ("sink_mass", "sink mass (first 4 tokens)", "magma", (0, 1)),
             ("frac_to_099", "fraction of KV blocks for 0.99 mass", "viridis", (0, 1))]
    for r, (col, title, cmap, lim) in enumerate(specs):
        mats = [per_genre.loc[g][col].unstack("head").values for g in GENRES if g in per_genre.index]
        mats.append(np.mean(mats, axis=0))
        for c, (mat, name) in enumerate(zip(mats, [*GENRES, "mean of genres"])):
            ax = axes[r, c]
            im = ax.imshow(mat, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto", interpolation="nearest")
            for x in range(4, 32, 4):
                ax.axvline(x - 0.5, color="w", lw=0.4, alpha=0.6)
            ax.set_title(f"{title} — {name}", fontsize=10)
            ax.set_xlabel("query head (groups of 4 share a KV head)")
            ax.set_ylabel("layer")
        fig.colorbar(im, ax=axes[r, :], shrink=0.9)
    fig.suptitle(f"Llama-3.1-8B, ctx {ctx // 1024}k, decode (median over 128 steps)", fontsize=13)
    save_fig(fig, f"{model}_{ctx}_layer_head_heatmaps")
    return table, by_layer


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------- blocks-needed hist

def bimodality_coefficient(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    skew = ((x - m) ** 3).mean() / s ** 3
    kurt = ((x - m) ** 4).mean() / s ** 4 - 3
    return (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def blocks_needed_hist(heads):
    out = {}
    configs = [("8b", 32768), ("8b", 16384), ("1b", 8192)]
    fig, axes = plt.subplots(2, len(configs), figsize=(18, 9), constrained_layout=True)
    bins = np.logspace(np.log10(1 / 700), 0, 46)
    for c, (model, ctx) in enumerate(configs):
        d = heads[(heads.model == model) & (heads.ctx == ctx)]
        if d.empty:
            continue
        nb = int(round(d["T"].mean() / 64))
        for r, (level, color) in enumerate((("0.9", "C0"), ("0.99", "C3"))):
            ax = axes[r, c]
            col = f"frac_blocks_to_{level}"
            per_head = d.groupby(["genre", "layer", "head"])[col].median()
            ax.hist(d[col], bins=bins, weights=np.full(len(d), 100 / len(d)), histtype="step", lw=1.5,
                    color="k", label="every (decode step, head)")
            ax.hist(per_head, bins=bins, weights=np.full(len(per_head), 100 / len(per_head)),
                    alpha=0.55, color=color, label="per-head median over 128 steps")
            lx = np.log10(per_head.clip(lower=1e-4))
            bc = bimodality_coefficient(lx)
            out[(model, ctx, level)] = {
                "bimodality_coef_per_head_log": float(bc),
                "share_heads_le_3_blocks": float((per_head * nb <= 3.5).mean()),
                "share_heads_lt_1pct": float((per_head < 0.01).mean()),
                "share_heads_1_10pct": float(((per_head >= 0.01) & (per_head < 0.1)).mean()),
                "share_heads_10_50pct": float(((per_head >= 0.1) & (per_head < 0.5)).mean()),
                "share_heads_ge_50pct": float((per_head >= 0.5).mean()),
            }
            ax.set_xscale("log")
            ax.set_title(f"{'Llama-3.1-8B' if model == '8b' else 'Llama-3.2-1B'}, {ctx // 1024}k ({nb} blocks): "
                         f"blocks for {level} of the mass\nbimodality coef. of per-head medians = {bc:.2f} "
                         f"(> 0.555 suggests bimodal)", fontsize=10)
            ax.set_xlabel("fraction of the context's 64-token blocks")
            ax.set_ylabel("% of samples")
            sec = ax.secondary_xaxis("top", functions=(lambda x, n=nb: x * n, lambda x, n=nb: x / n))
            sec.set_xlabel("blocks", fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    save_fig(fig, "blocks_needed_hist")
    t = pd.DataFrame(out).T
    t.index.names = ["model", "ctx", "mass"]
    return save_table(t, "blocks_needed_modes")


# ---------------------------------------------------------------- GQA union

def gqa_union(heads, kv):
    g = kv.groupby(["model", "ctx"])
    t = pd.DataFrame({
        "nblocks": (g["nblocks"].mean()).round(0),
        "mean_head_frac_0.9": (g["mean_head_blocks_to_0.9"].sum() / g["nblocks"].sum()),
        "union_frac_0.9": g["union_frac_blocks_to_0.9"].mean(),
        "mean_head_frac_0.99": (g["mean_head_blocks_to_0.99"].sum() / g["nblocks"].sum()),
        "max_head_frac_0.99": (g["max_head_blocks_to_0.99"].sum() / g["nblocks"].sum()),
        "union_frac_0.99": g["union_frac_blocks_to_0.99"].mean(),
    })
    t["union_over_mean_head_0.99"] = t["union_frac_0.99"] / t["mean_head_frac_0.99"]
    t["union_over_max_head_0.99"] = t["union_frac_0.99"] / t["max_head_frac_0.99"]
    gh = heads.groupby(["model", "ctx"])
    for tau in TAUS:
        t[f"skip_head_lt_{tau}"] = gh[f"frac_blocks_lt_{tau}"].mean()
        t[f"skip_union_lt_{tau}"] = g[f"union_frac_blocks_lt_{tau}"].mean()
        t[f"mass_lost_head_lt_{tau}"] = gh[f"mass_lost_lt_{tau}"].mean()
        t[f"mass_lost_union_lt_{tau}"] = gh[f"union_mass_lost_lt_{tau}"].mean()
    save_table(t, "gqa_union")

    # per layer, 8B 32k
    d = kv[(kv.model == "8b") & (kv.ctx == 32768)]
    if not d.empty:
        gl = d.groupby("layer")
        tl = pd.DataFrame({
            "mean_head_frac_0.99": gl["mean_head_blocks_to_0.99"].sum() / gl["nblocks"].sum(),
            "max_head_frac_0.99": gl["max_head_blocks_to_0.99"].sum() / gl["nblocks"].sum(),
            "union_frac_0.99": gl["union_frac_blocks_to_0.99"].mean(),
            "skip_union_lt_0.001": gl["union_frac_blocks_lt_0.001"].mean(),
        })
        hl = heads[(heads.model == "8b") & (heads.ctx == 32768)].groupby("layer")
        tl["skip_head_lt_0.001"] = hl["frac_blocks_lt_0.001"].mean()
        save_table(tl, "gqa_union_8b_32768_by_layer")
        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        x = tl.index.values
        ax.plot(x, tl["mean_head_frac_0.99"], "o-", label="per query head (mean of 4)")
        ax.plot(x, tl["max_head_frac_0.99"], "s-", label="max over the 4 query heads")
        ax.plot(x, tl["union_frac_0.99"], "^-", label="GQA union (blocks the KV head must load)")
        ax.set_xlabel("layer")
        ax.set_ylabel("fraction of KV blocks for 0.99 mass")
        ax.set_title("Llama-3.1-8B ctx 32k decode: per-head vs GQA-union block need (mean over steps, heads, genres)")
        ax.grid(alpha=0.3)
        ax.legend()
        save_fig(fig, "gqa_union_8b_32768_by_layer")
    return t


# ----------------------------------------------------------------- pre-exp

def preexp(heads, hist, edges):
    g = heads.groupby(["model", "ctx"])
    t = pd.DataFrame({"spearman_med": g["spearman_blockmax_vs_weight"].median(),
                      "spearman_p05": g["spearman_blockmax_vs_weight"].quantile(0.05)})
    for tau in TAUS:
        t[f"oracle_skip_{tau}"] = g[f"frac_blocks_lt_{tau}"].mean()
        t[f"oracle_mass_lost_{tau}"] = g[f"mass_lost_lt_{tau}"].mean()
        t[f"preexp_skip_{tau}"] = g[f"preexp_frac_skip_{tau}"].mean()
        t[f"preexp_mass_lost_{tau}"] = g[f"preexp_mass_lost_{tau}"].mean()
        t[f"preexp_false_skip_per_head_{tau}"] = g[f"preexp_false_skip_{tau}"].mean()
    save_table(t, "preexp_vs_oracle")

    # 2D histogram for 8B 32k pooled over genres and layers
    (g0, g1, gn), (w0, w1, wn) = edges
    gn, wn = int(gn), int(wn)
    keys = [k for k in hist if k[0] == "8b" and k[2] == 32768] or list(hist)[:1]
    H = sum(hist[k].sum(0) for k in keys).astype(float)            # [gap, logw]
    gap_c = g0 + (np.arange(gn) + 0.5) * (g1 - g0) / gn
    lw_c = w0 + (np.arange(wn) + 0.5) * (w1 - w0) / wn
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    im = ax.pcolormesh(np.linspace(g0, g1, gn + 1), np.linspace(w0, w1, wn + 1),
                       np.log10(H.T + 1), cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="log10(1 + #blocks)")
    gg = np.linspace(g0, g1, 200)
    ax.plot(gg, -gg / np.log(10), "w--", lw=1.2, label="W = exp(-gap)   (1 element, l = 1)")
    ax.plot(gg, np.log10(64) - gg / np.log(10), "w:", lw=1.2, label="W = 64·exp(-gap)  (upper bound, l ≥ 1)")
    # conditional quantiles of log10 W given the gap
    cdf = np.cumsum(H, axis=1)
    tot = cdf[:, -1:]
    ok = tot[:, 0] > 100
    for q, style in ((0.5, "r-"), (0.99, "r--"), (0.9999, "r:")):
        idx = (cdf >= q * tot).argmax(1)
        ax.plot(gap_c[ok], lw_c[idx][ok], style, lw=1.4, label=f"conditional {q:g}-quantile of W")
    for tau in (1e-4, 1e-3, 1e-2):
        ax.axhline(np.log10(tau), color="orange", lw=0.6, alpha=0.7)
        ax.axvline(np.log(1 / tau), color="orange", lw=0.6, alpha=0.7)
    ax.set_xlim(0, 30)
    ax.set_ylim(-14, 0)
    ax.set_xlabel("pre-exp gap = global max score − block max score (nats)")
    ax.set_ylabel("log10 block weight W (post-softmax)")
    ax.set_title("Does the block's max QK predict its weight? Llama-3.1-8B, 32k, decode, all layers/heads/genres\n"
                 "orange: thresholds τ ∈ {1e-4, 1e-3, 1e-2} for weight (horizontal) and pre-exp rule (vertical)")
    ax.legend(fontsize=8, loc="lower left")
    save_fig(fig, "preexp_hist2d_8b_32768")

    # trade-off: skip fraction vs mass lost, oracle vs pre-exp
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for (model, ctx), row in t.iterrows():
        o = [(row[f"oracle_skip_{tau}"], row[f"oracle_mass_lost_{tau}"]) for tau in TAUS]
        p = [(row[f"preexp_skip_{tau}"], row[f"preexp_mass_lost_{tau}"]) for tau in TAUS]
        ax = axes[0 if model == "8b" else 1]
        line, = ax.plot(*zip(*o), "o-", label=f"{ctx // 1024}k oracle (W < τ)")
        ax.plot(*zip(*p), "s--", color=line.get_color(), label=f"{ctx // 1024}k pre-exp (exp(M−m) < τ)")
    for ax, model in zip(axes, ("Llama-3.1-8B", "Llama-3.2-1B")):
        ax.set_yscale("log")
        ax.set_xlabel("fraction of KV blocks skipped (per head)")
        ax.set_ylabel("attention mass lost per head (mean)")
        ax.set_title(f"{model}: points are τ = 1e-4, 1e-3, 1e-2 (left to right)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    save_fig(fig, "preexp_vs_oracle_tradeoff")

    # spearman by layer, 8B 32k
    d = heads[(heads.model == "8b") & (heads.ctx == 32768)]
    if not d.empty:
        save_table(d.groupby("layer")["spearman_blockmax_vs_weight"].describe(percentiles=[0.05, 0.5]),
                   "preexp_spearman_8b_32768_by_layer")
    return t


def predictors(heads):
    """Extra: blocks needed when ranking by a cheaper predictor, and re-use of old selections."""
    if "blocks_to_0.99_by_quest" not in heads:
        return None
    g = heads.groupby(["model", "ctx"])
    t = pd.DataFrame({"nblocks": (g["T"].mean() / 64).round(0)})
    for level in ("0.9", "0.99"):
        oracle = g[f"blocks_to_{level}"].mean()
        t[f"oracle_{level}"] = oracle
        t[f"blockmax_{level}"] = g[f"blocks_to_{level}_by_blockmax"].mean()
        t[f"quest_{level}"] = g[f"blocks_to_{level}_by_quest"].mean()
        t[f"blockmax_over_oracle_{level}"] = t[f"blockmax_{level}"] / oracle
        t[f"quest_over_oracle_{level}"] = t[f"quest_{level}"] / oracle
    t["quest_slack_nats_med"] = g["quest_slack_median"].median()
    for lag in (1, 16):
        t[f"reuse_mass_lag{lag}_mean"] = g[f"reuse_mass_lag{lag}"].mean()
        t[f"reuse_mass_lag{lag}_p05"] = g[f"reuse_mass_lag{lag}"].quantile(0.05)
        t[f"reuse_frac_blocks_lag{lag}"] = g[f"reuse_frac_blocks_lag{lag}"].mean()
    t["frac_blocks_to_0.99_mean"] = g["frac_blocks_to_0.99"].mean()
    return save_table(t, "extra_predictors_reuse")


# ----------------------------------------------------------------- ablation

def load_ablation():
    rows = []
    for f in sorted((RES / "ablate").glob("*.summary.json")):
        s = json.loads(f.read_text())
        for name, c in s["conditions"].items():
            rows.append({"model": s["model"], "genre": s["genre"], "ctx": s["ctx"], **c})
    return pd.DataFrame(rows)


def ablation(ab: pd.DataFrame):
    if ab.empty:
        return None
    steps = 128
    ab = ab.copy()
    ab["div_tf"] = ab["first_div_tf"].fillna(steps)
    ab["div_free"] = ab["first_div_free"].fillna(steps)
    keep = ["kl_mean", "kl_median", "kl_p95", "kl_max", "top1_agree", "div_tf", "div_free",
            "frac_blocks_skipped", "mass_dropped_mean"]
    per_doc = ab.set_index(["model", "ctx", "genre", "condition"])[keep]
    save_table(per_doc, "ablation_per_doc")
    g = ab.groupby(["model", "ctx", "condition"], sort=False)
    agg = pd.DataFrame({
        "kl_mean": g["kl_mean"].mean(),
        "kl_p95": g["kl_p95"].mean(),
        "kl_max": g["kl_max"].max(),
        "top1_agree": g["top1_agree"].mean(),
        "first_div_min": g["div_tf"].min(),
        "first_div_by_genre": g["div_tf"].agg(lambda s: "/".join(str(int(v)) for v in s)),
        "frac_blocks_skipped": g["frac_blocks_skipped"].mean(),
        "mass_dropped_mean": g["mass_dropped_mean"].mean(),
    })
    save_table(agg, "ablation_summary")

    # check: teacher-forced divergence step == free-running divergence step
    consistent = (ab["div_tf"] == ab["div_free"]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    configs = sorted(ab[["model", "ctx"]].drop_duplicates().itertuples(index=False), key=lambda r: (r.model, r.ctx))
    for ax, model in zip(axes, ("8b", "1b")):
        for ctx in sorted({c.ctx for c in configs if c.model == model}):
            d = agg.loc[(model, ctx)]
            floor = d.loc[[c for c in d.index if c.startswith("stock_")], "kl_mean"].max()
            ax.axhline(floor, ls=":", color="gray")
            ax.text(0.01, floor * 1.1, f"bf16 stock-kernel noise floor ({ctx // 1024}k)", fontsize=7, color="gray")
            for crit, scope, marker in (("weight", "head", "o"), ("weight", "union", "^"),
                                        ("preexp", "head", "s"), ("preexp", "union", "D"),
                                        ("mass", "head", "x"), ("mass", "union", "P")):
                levels = ("0.999", "0.99", "0.9") if crit == "mass" else ("0.0001", "0.001", "0.01")
                names = [f"{crit}_{tau}_{scope}" for tau in levels]
                dd = d.loc[[n for n in names if n in d.index]]
                if dd.empty:
                    continue
                ax.plot(dd["frac_blocks_skipped"], dd["kl_mean"], marker=marker,
                        ls={"weight": "-", "preexp": "--", "mass": ":"}[crit],
                        label=f"{ctx // 1024}k {crit} / {scope}")
        ax.set_yscale("log")
        ax.set_xlabel("fraction of KV blocks skipped (mean over layers, heads, steps)")
        ax.set_ylabel("KL(ref || masked) per decode step, mean over steps and genres")
        ax.set_title(f"{'Llama-3.1-8B' if model == '8b' else 'Llama-3.2-1B'}: τ = 1e-4, 1e-3, 1e-2 left→right; "
                     f"mass (top-p) 0.999, 0.99, 0.9 left→right")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    save_fig(fig, "ablation_tradeoff")
    return agg, consistent


# -------------------------------------------------------------- oddities

def oddities(heads, model="8b", ctx=32768):
    d = heads[(heads.model == model) & (heads.ctx == ctx)]
    if d.empty:
        return None
    ph = d.groupby(["layer", "head"]).agg(
        entropy_norm=("entropy_norm", "median"),
        sink_mass=("sink_mass", "median"),
        local_mass=("local_mass", "median"),
        frac_to_099=("frac_blocks_to_0.99", "median"),
        frac_to_09=("frac_blocks_to_0.9", "median"),
        max_prob=("max_prob", "median"),
        spearman=("spearman_blockmax_vs_weight", "median"),
        entropy_genre_spread=("entropy_norm", lambda s: 0.0),
    )
    by_genre = d.groupby(["layer", "head", "genre"])["frac_blocks_to_0.99"].median().unstack("genre")
    ph["frac_to_099_genre_range"] = by_genre.max(axis=1) - by_genre.min(axis=1)
    ph = ph.drop(columns="entropy_genre_spread")
    out = {
        "most_uniform": ph.sort_values("entropy_norm", ascending=False).head(12),
        "most_blocks_needed": ph.sort_values("frac_to_099", ascending=False).head(12),
        "low_sink_high_entropy": ph[(ph.sink_mass < 0.05)].sort_values("entropy_norm", ascending=False).head(12),
        "local_heads": ph.sort_values("local_mass", ascending=False).head(12),
        "lowest_spearman": ph.sort_values("spearman").head(12),
        "genre_dependent": ph.sort_values("frac_to_099_genre_range", ascending=False).head(12),
    }
    for k, v in out.items():
        save_table(v, f"odd_{model}_{ctx}_{k}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, default=RES / "collect")
    args = ap.parse_args()
    heads, kv, hist, edges, meta = load_collect(args.collect)
    print(f"collect: {heads.groupby(['model', 'ctx', 'genre']).ngroups} runs, {len(heads)} head rows")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(overview(heads, kv).round(3))
    print(scaling(heads).round(3))
    layer_head(heads)
    print(blocks_needed_hist(heads).round(3))
    print(gqa_union(heads, kv).round(3).T)
    print(preexp(heads, hist, edges).round(4).T)
    extra = predictors(heads)
    if extra is not None:
        print(extra.round(3).T)
    oddities(heads)
    ab = load_ablation()
    res = ablation(ab)
    if res is not None:
        agg, consistent = res
        print(agg.round(4))
        print(f"teacher-forced vs free-running first-divergence step agree in {consistent:.1%} of runs")
    gens = {f"{k[0]}_{k[1]}_{k[2]}": v["generated"] for k, v in meta.items()}
    (RES / "generated_texts.json").write_text(json.dumps(gens, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
