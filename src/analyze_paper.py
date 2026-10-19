"""Tables and figures for the paper that the stage scripts do not already write.

  paper_ablation_topp   stage 1's top-p masking ablation (keep the fewest blocks holding a share p of
                        each head's mass): the runs exist in results/ablate/*.mass.summary.json, but
                        the stage-1 tables were made before they finished. Aggregated as
                        analyze.ablation aggregates ablation_summary (mean over the three genres),
                        with KL in multiples of stock bf16 SDPA's on the same model and context.

Figures (vector PDF at ACL column or text width; each also writes the values it plots to a
results/tables/paper_fig*.csv, so the numbers quoted from a figure come from a table):
  paper_fig1_jumps      Llama-3.1-8B, prose, 32k. (a) Under a 32-step timer, the KV heads whose stored
                        set newly stops covering 95% of the mass at each step (stage 3's jump count:
                        both this and the previous step stale, layers > 0). (b) Per-step KL from the
                        fp32 reference: the timer, the stage-4 event trigger, stock bf16 SDPA.
  paper_fig3_blind      Llama-3.1-8B, prose, 16k, the induction head (2, 5): the mass its stored set
                        really misses against the pre-pass estimate the trigger acts on, (a) trigger
                        alone, (b) with stage 4's peak-hold correction c.

Tables for the paper's statistics (bootstrap intervals, exact paired tests, timers at equal bytes, the
lab's KL per document) and for RULER (ruler_tables, ruler_detail) are described at their functions.

Writes results/tables/paper_*.{csv,md} and results/figures/paper_*.pdf.
Input: results/ablate/*.summary.json, results/reuse/8b_*.{steps,heads}.parquet and *.summary.json,
results/lab/s4_*.json, results/ruler/8b_32768_main.json, the stage tables in results/tables.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAB = RES / "tables"
FIG = RES / "figures"
REUSE = RES / "reuse"
POLICY = "estp_delta0.05_age32_cal0.05w32a8peak_aix0.1"          # the stage-4 policy
COL, FULL = 3.15, 6.3                                              # ACL column and text width, inches
# One color per method in every figure (the dataviz reference palette's categorical slots, in order).
COLORS = {"policy": "#2a78d6", "questp": "#eb6834", "quest_kv": "#1baf7a", "timer": "#eda100",
          "static": "#e87ba4", "oracle": "#6250d6", "list": "#e34948", "sdpa": "#8c8b87"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.color": "#e2e2df", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save_table(df: pd.DataFrame, name: str, floatfmt=".3g"):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv")
    (TAB / f"{name}.md").write_text(df.to_markdown(floatfmt=floatfmt))
    return df


def save_fig(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def ablation_topp():
    rows = []
    for f in sorted((RES / "ablate").glob("*.summary.json")):
        s = json.loads(f.read_text())
        for name, c in s["conditions"].items():
            rows.append({"model": s["model"], "genre": s["genre"], "ctx": s["ctx"], **c})
    ab = pd.DataFrame(rows)
    g = ab.groupby(["model", "ctx", "condition"])
    agg = pd.DataFrame({"kl_mean": g["kl_mean"].mean(), "top1_agree": g["top1_agree"].mean(),
                        "frac_blocks_skipped": g["frac_blocks_skipped"].mean(),
                        "mass_dropped_mean": g["mass_dropped_mean"].mean(), "genres": g.size()})
    noise = agg.xs("stock_sdpa_bf16", level="condition")["kl_mean"]
    agg["kl_x_noise"] = agg["kl_mean"] / noise.reindex(agg.index.droplevel("condition")).to_numpy()
    keep = [c for c in agg.index.get_level_values("condition").unique()
            if c.startswith("mass_") or c in ("stock_sdpa_bf16", "weight_0.0001_union", "weight_0.001_union")]
    t = agg[agg.index.get_level_values("condition").isin(keep)]
    t = t[["frac_blocks_skipped", "mass_dropped_mean", "kl_x_noise", "kl_mean", "top1_agree", "genres"]]
    return save_table(t.sort_index(), "paper_ablation_topp")


def reuse_lag():
    """Stage 1's re-use of old selections (analyze.predictors, whose table extra_predictors_reuse was
    never written): the mass that a query head's 0.99 block set from 1 or 16 steps earlier still covers
    (with the 2 newest blocks), mean and 5th percentile over heads, steps and genres."""
    cols = ["model", "ctx", "T", "reuse_mass_lag1", "reuse_frac_blocks_lag1", "reuse_mass_lag16",
            "reuse_frac_blocks_lag16", "frac_blocks_to_0.99"]
    heads = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((RES / "collect_v2").glob("*.heads.parquet"))])
    g = heads.groupby(["model", "ctx"])
    t = pd.DataFrame({"nblocks": (g["T"].mean() / 64).round(0)})
    for lag in (1, 16):
        t[f"reuse_mass_lag{lag}_mean"] = g[f"reuse_mass_lag{lag}"].mean()
        t[f"reuse_mass_lag{lag}_p05"] = g[f"reuse_mass_lag{lag}"].quantile(0.05)
        t[f"reuse_frac_blocks_lag{lag}"] = g[f"reuse_frac_blocks_lag{lag}"].mean()
    t["frac_blocks_to_0.99_mean"] = g["frac_blocks_to_0.99"].mean()
    return save_table(t, "paper_reuse_lag")


def fig_jumps(doc="prose", ctx=32768):
    h = pd.read_parquet(REUSE / f"8b_{doc}_{ctx}.diag.heads.parquet",
                        columns=["condition", "step", "layer", "kv_head", "covered_stale_min"])
    g = h[(h.condition == "timer_N32") & (h.layer > 0)]
    lost = g.assign(lost=g.covered_stale_min < 0.95).pivot_table(
        index="step", columns=["layer", "kv_head"], values="lost").astype(bool)
    steps = [t for t in range(2, len(lost)) if lost.index[t] % 32 >= 2]          # as analyze_stage3.jumps
    newly = pd.Series([int((lost.iloc[t] & ~lost.iloc[t - 1]).sum()) for t in steps],
                      index=[int(lost.index[t]) for t in steps], name="kv_heads_newly_lost")
    s2 = pd.read_parquet(REUSE / f"8b_{doc}_{ctx}.steps.parquet")
    s4 = pd.read_parquet(REUSE / f"8b_{doc}_{ctx}.s4q.steps.parquet")
    kl = pd.DataFrame({"sdpa": s2[s2.condition == "stock_sdpa_bf16"].set_index("step").kl,
                       "timer_N32": s2[s2.condition == "timer_N32"].set_index("step").kl,
                       "policy": s4[s4.condition == POLICY].set_index("step").kl,
                       "sdpa_s4q": s4[s4.condition == "stock_sdpa_bf16"].set_index("step").kl})
    t = kl.join(newly, how="left")
    t["sparse_kv_heads"] = lost.shape[1]
    save_table(t, "paper_fig1_jumps")

    fig, (a, b) = plt.subplots(2, 1, figsize=(COL, 2.3), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.38})
    a.bar(newly.index, newly.values, width=0.8, color=COLORS["sdpa"], linewidth=0)
    peak = int(newly.idxmax())
    a.annotate(f"{newly.max()} of {lost.shape[1]}", (peak, newly.max()), xytext=(4, -2),
               textcoords="offset points", fontsize=7, va="top")
    a.set_ylabel("KV heads losing\ntheir blocks")
    for t0 in range(0, len(kl), 32):
        b.axvline(t0, color=COLORS["timer"], lw=0.6, alpha=0.5, zorder=0)
    floor = 1e-8
    b.plot(kl.index, kl.sdpa.clip(lower=floor), color=COLORS["sdpa"], lw=0.8, label="stock bf16 SDPA")
    b.plot(kl.index, kl.timer_N32.clip(lower=floor), color=COLORS["timer"], lw=0.9, label="timer, every 32 steps")
    b.plot(kl.index, kl.policy.clip(lower=floor), color=COLORS["policy"], lw=0.9, label="event trigger (ours)")
    b.set_yscale("log")
    b.set_ylim(floor, None)
    b.set_ylabel("KL from fp32 (nats)")
    b.set_xlabel("decode step")
    b.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, handlelength=1.0, columnspacing=0.8,
             borderaxespad=0.1, fontsize=6.5)
    a.grid(axis="x", visible=False)
    b.grid(axis="x", visible=False)
    save_fig(fig, "paper_fig1_jumps")
    return t


MODEL_NAMES = {"8b": "Llama-3.1-8B", "qwen3-8b": "Qwen3-8B", "qwen2.5-7b": "Qwen2.5-7B", "mistral-7b": "Mistral-7B"}


def fig_reads(ctx=32768):
    """(a) Llama-3.1-8B: per-head median (over 128 steps) share of the blocks holding 0.9 and 0.99 of a
    query head's mass, the three genres (stage 1, results/collect_v2, as in blocks_needed_hist);
    (b, c) four models: KV-head blocks for 0.99 and the sink mass by relative depth (stage 4's map)."""
    h = pd.concat([pd.read_parquet(f, columns=["genre", "layer", "head", "frac_blocks_to_0.9", "frac_blocks_to_0.99"])
                   for f in sorted((RES / "collect_v2").glob(f"8b_*_{ctx}.heads.parquet"))])
    per_head = h.groupby(["genre", "layer", "head"])[["frac_blocks_to_0.9", "frac_blocks_to_0.99"]].median()
    rows = []
    for model in MODEL_NAMES:
        k = pd.concat([pd.read_parquet(f, columns=["layer", "union_frac_blocks_to_0.99"])
                       for f in sorted((RES / "collect").glob(f"{model}_*_{ctx}.kv.parquet"))])
        s = pd.concat([pd.read_parquet(f, columns=["layer", "sink_mass"])
                       for f in sorted((RES / "collect").glob(f"{model}_*_{ctx}.heads.parquet"))])
        L = int(k.layer.max()) + 1
        t = pd.DataFrame({"kv_blocks_0.99": k.groupby("layer")["union_frac_blocks_to_0.99"].mean(),
                          "sink_mass_median": s.groupby("layer")["sink_mass"].median()})
        t["model"], t["depth"] = MODEL_NAMES[model], t.index / (L - 1)
        rows.append(t.reset_index())
    depth = pd.concat(rows, ignore_index=True)
    save_table(per_head, "paper_fig2a_blocks_per_head")
    save_table(depth.set_index(["model", "layer"]), "paper_fig2bc_depth")

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.65), gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.38})
    a, b, c = axes
    bins = np.logspace(np.log10(1 / 600), 0, 40)
    for col, color, label in (("frac_blocks_to_0.9", COLORS["sdpa"], "90% of the mass"),
                              ("frac_blocks_to_0.99", COLORS["policy"], "99% of the mass")):
        v = per_head[col].clip(lower=bins[0])
        a.hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)), color=color, alpha=0.75, label=label)
    a.set_xscale("log")
    a.set_xlabel("share of the context's blocks")
    a.set_ylabel("% of query heads")
    a.set_title("(a) Llama-3.1-8B, 32k: blocks for", loc="left", pad=3)
    a.legend(loc="upper right", handlelength=1.0, borderaxespad=0.1)
    a.grid(axis="x", visible=False)
    order = list(MODEL_NAMES.values())
    palette = [COLORS["policy"], COLORS["questp"], COLORS["quest_kv"], COLORS["timer"]]
    for name, color in zip(order, palette):                   # dashed: the Qwen family (fact 3)
        d, ls = depth[depth.model == name], "--" if name.startswith("Qwen") else "-"
        b.plot(d.depth, d["kv_blocks_0.99"], color=color, lw=1.0, ls=ls, label=name)
        c.plot(d.depth, d["sink_mass_median"], color=color, lw=1.0, ls=ls, label=name)
    b.set_title("(b) KV-head blocks for 99%", loc="left", pad=3)
    c.set_title("(c) mass on the first 4 tokens", loc="left", pad=3)
    b.set_ylim(0, 1)
    c.set_ylim(0, 1)
    for ax in (b, c):
        ax.set_xlabel("relative depth (layer / last layer)")
    b.legend(loc="lower center", ncol=2, handlelength=1.6, columnspacing=0.6, borderaxespad=0.1, fontsize=6.5)
    save_fig(fig, "paper_fig2_reads")
    return per_head, depth


def fig_blind(doc="prose", ctx=16384, head=(2, 5)):
    cols = ["condition", "step", "layer", "kv_head", "covered_stale_min", "miss_p", "calib", "refresh", "auto_ix"]
    runs = {"alone": (f"8b_{doc}_{ctx}.s4a.heads.parquet", "estp_delta0.05_age32"),
            "calibrated": (f"8b_{doc}_{ctx}.s4p.heads.parquet", POLICY)}
    parts = []
    for key, (f, cond) in runs.items():
        h = pd.read_parquet(REUSE / f, columns=cols)
        h = h[(h.condition == cond) & (h.layer == head[0]) & (h.kv_head == head[1])].set_index("step")
        parts.append(pd.DataFrame({f"{key}_real_miss": 1.0 - h.covered_stale_min, f"{key}_estimate": h.miss_p,
                                   f"{key}_c": h.calib, f"{key}_refresh": h.refresh, f"{key}_index": h.auto_ix}))
    t = pd.concat(parts, axis=1)
    save_table(t, "paper_fig3_blind")

    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.2), sharex=True, sharey=True)
    for ax, key, title in zip(axes, ("alone", "calibrated"),
                              ("trigger alone", "with the peak-hold correction (stage 4)")):
        r = t[f"{key}_refresh"] > 0
        ax.plot(t.index, t[f"{key}_real_miss"], color=COLORS["list"], lw=0.9, label="mass the stored set misses")
        ax.plot(t.index, t[f"{key}_estimate"], color=COLORS["policy"], lw=0.9, label="pre-pass estimate")
        if key == "calibrated":
            ax.plot(t.index, t[f"{key}_estimate"] + t[f"{key}_c"], color=COLORS["policy"], lw=0.9, ls="--",
                    label="estimate + correction $c$")
        ax.scatter(t.index[r], np.full(r.sum(), -0.04), marker="|", s=10, color="#52514e", lw=0.6,
                   label="refresh", clip_on=False)
        ax.axhline(0.05, color="#52514e", lw=0.5, ls=":")
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylim(-0.06, 1.0)
        ax.set_ylabel("share of the head's mass")
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("decode step")
    axes[1].legend(loc="upper right", fontsize=6.5, handlelength=1.2, borderaxespad=0.2)
    save_fig(fig, "paper_fig3_blind")
    return t


def boot(x, stat="mean", n=10000, seed=0):
    """Point estimate and percentile-bootstrap 95% interval over documents (resampled with replacement),
    worst document and count. With three documents the interval is close to their range."""
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    if stat == "gmean":
        point, s = np.exp(np.log(x).mean()), np.exp(np.log(x)[idx].mean(1))
    else:
        point, s = x.mean(), x[idx].mean(1)
    lo, hi = np.quantile(s, [0.025, 0.975])
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "worst": x.max(), "docs": len(x)}


EQUAL_ROWS = {("refresh", "p0.99"): "refresh + calibration (ours)",
              ("refresh_s3", "p0.99"): "refresh + 8 hand-picked exact heads",
              ("topk", "p0.99"): "per-step oracle",
              ("quest_s4best", "p0.99"): "Quest, pages per KV head",
              ("quest_s4kickoff", "p0.99"): "Quest, 16-token bounds, 64-token loads",
              ("questp", "12288"): "Quest as published (12288 tokens)",
              ("static", "p0.99"): "static",
              ("vskip1", "-"): "single-pass V-skip",
              ("refresh", "p0.2"): "refresh + calibration at top-p 0.2",
              ("quest_s4best", "questp2048"): "Quest per KV head at Quest-2048's bytes",
              ("questp", "2048"): "Quest as published (2048 tokens)",
              ("topk", "questp2048"): "per-step oracle at Quest-2048's bytes"}


def ci_tables():
    """Bootstrap intervals over documents for the paper's tables: equal bytes (geometric mean over six
    documents, stage5_budget_points_by_doc), calibration (mean over three, stage4_calib_by_doc) and the
    four models (mean over three, the stage-4 run summaries)."""
    sys.path.insert(0, str(ROOT / "src"))
    import analyze_stage4 as s4
    p = pd.read_csv(TAB / "stage5_budget_points_by_doc.csv").set_index(["family", "point"])
    pts = pd.read_csv(TAB / "stage5_budget_points.csv")
    pts.columns = ["family", "point"] + list(pts.columns[2:])
    pts = pts.set_index(["family", "point"])
    rows = []
    for key, label in EQUAL_ROWS.items():
        r = boot(p.loc[key].dropna().to_numpy(), "gmean")
        rows.append({"method": label, "bytes": pts.loc[key, "bytes_total"], **r})
    eq = save_table(pd.DataFrame(rows).set_index("method"), "paper_ci_equal")
    ours = p.loc[("refresh", "p0.99")]                            # paired: the same six documents
    rows = []
    for key, label in list(EQUAL_ROWS.items())[1:8]:
        ratio = (p.loc[key] / ours).dropna()
        rows.append({"method": label, "docs_worse_than_ours": int((ratio > 1).sum()),
                     **{f"ratio_{k}": v for k, v in boot(ratio.to_numpy(), "gmean").items() if k != "worst"}})
    save_table(pd.DataFrame(rows).set_index("method"), "paper_ci_equal_paired")

    c = pd.read_csv(TAB / "stage4_calib_by_doc.csv")
    c = c[c.condition.isin(s4.KEY_VARIANTS)]
    rows = []
    for (model, st, cond), d in c.groupby(["model", "set", "condition"]):
        rows.append({"model": model, "set": st, "variant": s4.KEY_VARIANTS[cond],
                     **boot(d.kl_x_noise.to_numpy()), "bytes": d.bytes_total.mean()})
    cal = save_table(pd.DataFrame(rows).set_index(["model", "set", "variant"]), "paper_ci_calib")

    rows = []
    for model in MODEL_NAMES:
        sm = s4.summaries(["diag", "s4a", "s4b", "s4p"], model=model)
        sm = sm[(sm.ctx == 32768) & sm.doc.isin(s4.STAGE1_DOCS)].drop_duplicates(["doc", "condition"], keep="last")
        for cond, label in (("timer_N32", "timer N=32"), ("estp_delta0.05_age32", "trigger alone"),
                            (s4.FINAL, "trigger + calibration")):
            d = sm[sm.condition == cond]
            if not d.empty:
                rows.append({"model": MODEL_NAMES[model], "condition": label, **boot(d.kl_x_noise.to_numpy())})
    mod = save_table(pd.DataFrame(rows).set_index(["model", "condition"]), "paper_ci_models")
    return eq, cal, mod


def lab_kl():
    """Table 3's KL ratios per document (stage 4's lab runs, eight documents at 32k and 64k): the lab's
    x_noise (as in stage4_gate_summary), the worst document and a bootstrap interval of the per-document mean."""
    rows = []
    for ctx in (32768, 65536):
        d = json.loads((RES / "lab" / f"s4_{ctx}.json").read_text())
        for cond, label in (("refresh_cal", "ours"), ("refresh_fused", "hand-tuned"), ("vskip1", "single-pass V-skip"),
                            ("dense", "dense kernel")):
            k = d["kl"][cond]
            by_doc = np.asarray(k["x_noise_by_doc"], dtype=float)
            b = boot(by_doc)
            rows.append({"ctx": ctx, "condition": label, "x_noise": k["x_noise"], "doc_mean": b["point"],
                         "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "worst": b["worst"], "docs": b["docs"]})
    return save_table(pd.DataFrame(rows).set_index(["ctx", "condition"]), "paper_lab_kl")


def timer_equal():
    """Timers against the event trigger at about the same bytes (Llama-3.1-8B, the three stage-1 documents,
    16k and 32k): timers, a per-head query-drift trigger and an oracle trigger (refresh when the stale set's
    exact coverage falls below 0.98; needs all of K) from stage 2 (stage2_a1_reuse), the event trigger alone,
    with the stage-3 head list and with stage 4's calibration from stage 4 (stage4_calib_main(_bytes)).
    Stage-2 bytes carry no metadata (timers and the oracle have none)."""
    s2 = pd.read_csv(TAB / "stage2_a1_reuse.csv")
    s2.columns = ["ctx", "condition"] + list(s2.columns[2:])
    rows = []
    names = {"timer_N4": "timer, every 4 steps", "timer_N8": "timer, every 8 steps", "timer_N16": "timer, every 16 steps",
             "timer_N32": "timer, every 32 steps", "drift_delta0.53_age32": "query-drift trigger (per head)",
             "oracle_theta0.98_age32": "oracle trigger (coverage < 0.98)"}
    for _, r in s2[s2.condition.isin(names)].iterrows():
        rows.append({"ctx": int(r.ctx), "method": names[r.condition], "kl_x_noise": r.KL_x_noise_mean,
                     "worst": r.KL_x_noise_max, "bytes": 1.0 - r.KV_bytes_skip})
    km = pd.read_csv(TAB / "stage4_calib_main.csv", index_col=0)
    kb = pd.read_csv(TAB / "stage4_calib_main_bytes.csv", index_col=0)
    for variant, label in (("trigger alone (no list, delta 0.05)", "event trigger alone"),
                           ("+ 8 always-exact heads (stage 3, Llama)", "event trigger + 8-head list"),
                           ("peak-hold + warm-up + auto token index (stage 4)", "event trigger + calibration (ours)")):
        for ctx in (16384, 32768):
            col = f"Llama-3.1-8B: stage-1 docs {ctx // 1024}k"
            mean, worst = km.loc[variant, col].replace("(", "").replace(")", "").split()
            rows.append({"ctx": ctx, "method": label, "kl_x_noise": float(mean), "worst": float(worst),
                         "bytes": float(kb.loc[variant, col])})
    return save_table(pd.DataFrame(rows).set_index(["ctx", "method"]).sort_index(), "paper_timer_equal")


def exact_tests(log_ratio):
    """Two-sided exact Wilcoxon signed-rank and sign-test p-values for paired log ratios (small n)."""
    x = np.asarray(log_ratio, dtype=float)
    n = len(x)
    ranks = pd.Series(np.abs(x)).rank().to_numpy()
    w = ranks[x > 0].sum()
    signs = np.array(np.meshgrid(*[[0, 1]] * n)).reshape(n, -1).T
    dist = (signs * ranks).sum(1)
    mu = ranks.sum() / 2
    p_w = float(np.mean(np.abs(dist - mu) >= abs(w - mu) - 1e-12))
    k = int((x > 0).sum())
    from math import comb
    tail = sum(comb(n, i) for i in range(0, n + 1) if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n
    return p_w, float(tail)


def equal_tests():
    """Per-document paired ratios at the operating point (method / ours, six documents) with exact tests."""
    p = pd.read_csv(TAB / "stage5_budget_points_by_doc.csv").set_index(["family", "point"])
    ours = p.loc[("refresh", "p0.99")]
    rows = []
    for key, label in list(EQUAL_ROWS.items())[1:8]:
        r = (p.loc[key] / ours).dropna()
        p_w, p_s = exact_tests(np.log(r.to_numpy()))
        rows.append({"method": label, **{f"ratio_{d}": v for d, v in r.items()}, "median": float(np.median(r)),
                     "worse_docs": int((r > 1).sum()), "wilcoxon_p": p_w, "sign_p": p_s})
    return save_table(pd.DataFrame(rows).set_index("method"), "paper_equal_tests")


def trigger_fail_heads(ctx=16384):
    """Where the trigger alone misses (Llama, stage-1 documents): head-steps whose attended set covers less than
    80% of the head's mass, and how many of them belong to the induction head (2, 5)."""
    rows = []
    for doc in ("prose", "code", "dialog"):
        h = pd.read_parquet(REUSE / f"8b_{doc}_{ctx}.s4a.heads.parquet",
                            columns=["condition", "step", "layer", "kv_head", "covered_min"])
        h = h[(h.condition == "estp_delta0.05_age32") & (h.layer > 0)]
        low = h[h.covered_min < 0.8]
        rows.append({"doc": doc, "ctx": ctx, "head_steps_cov_lt_0.8": len(low),
                     "of_head_2_5": int(((low.layer == 2) & (low.kv_head == 5)).sum())})
    return save_table(pd.DataFrame(rows).set_index("doc"), "paper_trigger_fail_heads")


def carry_list_reference():
    """Stage 5 compared carrying the calibration across requests (16 (document, order) pairs: nine Llama
    documents at 32k in two orders, the first of each chain excluded) with the stage-3 hand-picked list; the
    list's mean over the same 16 entries, from its per-document stage-4 runs (order does not affect it)."""
    c = pd.read_csv(TAB / "stage4_calib_by_doc.csv")
    c = c[(c.model == "8b") & (c.ctx == 32768) & (c.condition == "estp_delta0.05_age32_dh8")]
    per_doc = c.drop_duplicates("doc", keep="last").set_index("doc")
    fwd = ["prose", "code", "dialog", "prose2", "code2", "dialog2", "prose3", "code3", "dialog3"]
    entries = fwd[1:] + fwd[::-1][1:]
    t = pd.DataFrame({"pairs": [len(entries)], "kl_mean": [per_doc.loc[entries, "kl_x_noise"].mean()],
                      "kl_worst": [per_doc.loc[entries, "kl_x_noise"].max()],
                      "bytes": [per_doc.loc[entries, "bytes_total"].mean()]}, index=["8 hand-picked exact heads"])
    return save_table(t, "paper_carry_list")


RULER_TASKS = ["niah_multikey_2", "niah_multivalue", "vt", "fwe", "qa_1"]
RULER_ROWS = {                                                    # condition -> label, in table order
    "ref": "exact (fp32)", "sdpa": "stock bf16 SDPA",
    POLICY: "refresh + calibration (ours)",
    "estp_delta0.05_age32_ixu1_cal0.05w32a8peak_aix0.1": "  + head (2,5) latched to the index",
    f"quest16softnative_match-{POLICY}": "Quest per KV head, our bytes",
    f"topk_match-{POLICY}": "per-step oracle, our bytes",
    "questp2048": "Quest as published, 2048 tokens",
    "quest16softnative_match-questp2048": "Quest per KV head, Quest's bytes",
    "topk_match-questp2048": "per-step oracle, Quest's bytes",
    "estp_delta0.84_age32_cal0.05w32a8peak_aix0.1_p0.2": "refresh + calibration, top-p 0.2"}


def ruler_tables(path=RES / "ruler" / "8b_32768_main.json", n_boot=10000):
    """RULER at 32k: score per task and condition (RULER's metric, x100), the average over tasks with a
    bootstrap interval over samples (resampled within each task), the paired difference from the exact
    reference, K+V bytes and the KL ratio on the question tokens (geometric mean over samples)."""
    runs = json.loads(Path(path).read_text())["runs"]
    recs = []
    for r in runs:
        recs.append({"task": r["task"], "sample": r["sample"], "condition": "ref", "score": r["ref"]["score"],
                     "bytes": 1.0, "kl_q": np.nan})
        recs.append({"task": r["task"], "sample": r["sample"], "condition": "sdpa", "score": r["sdpa"]["score"],
                     "bytes": 1.0, "kl_q": 1.0})
        for name, c in r["conditions"].items():
            recs.append({"task": r["task"], "sample": r["sample"], "condition": name, "score": c["score"],
                         "bytes": c["bytes_total"], "kl_q": c["kl_q_x_noise"]})
    d = pd.DataFrame(recs)
    tasks = [t for t in RULER_TASKS if t in set(d.task)]
    full = d.pivot_table(index=["task", "sample"], columns="condition", values="score")
    per_task = d.groupby(["condition", "task"]).score.mean().unstack()[tasks] * 100
    rng = np.random.default_rng(0)
    groups = {t: full.loc[t] for t in tasks}
    draws = [{t: rng.integers(0, len(g), len(g)) for t, g in groups.items()} for _ in range(n_boot)]

    def avg(cond):                                             # mean over tasks of the per-task mean
        return np.mean([groups[t][cond].to_numpy().mean() for t in tasks])

    rows = []
    for cond, label in RULER_ROWS.items():
        if cond not in full.columns:
            continue
        b = np.array([np.mean([groups[t][cond].to_numpy()[ix[t]].mean() for t in tasks]) for ix in draws])
        diff = np.array([np.mean([(groups[t][cond].to_numpy() - groups[t]["ref"].to_numpy())[ix[t]].mean()
                                  for t in tasks]) for ix in draws])
        sub = d[d.condition == cond]
        kl = sub.kl_q.dropna()
        rows.append({"condition": label, "bytes": sub.bytes.mean(), **per_task.loc[cond].to_dict(),
                     "average": avg(cond) * 100, "avg_ci_lo": np.quantile(b, 0.025) * 100,
                     "avg_ci_hi": np.quantile(b, 0.975) * 100,
                     "diff_vs_exact": (avg(cond) - avg("ref")) * 100,
                     "diff_ci_lo": np.quantile(diff, 0.025) * 100, "diff_ci_hi": np.quantile(diff, 0.975) * 100,
                     "kl_q_gmean": float(np.exp(np.log(kl.clip(lower=1e-6)).mean())) if len(kl) else np.nan,
                     "samples_per_task": int(full.loc[tasks[0]].shape[0])})
    t = save_table(pd.DataFrame(rows).set_index("condition"), "paper_ruler")
    by_task_bytes = d.groupby(["condition", "task"]).bytes.mean().unstack()[tasks]
    save_table(by_task_bytes.rename(index={k: v for k, v in RULER_ROWS.items()}), "paper_ruler_bytes")
    return t


RULER_PAIRS = [(POLICY, f"quest16softnative_match-{POLICY}"),                 # (method, against), same samples
               ("estp_delta0.05_age32_ixu1_cal0.05w32a8peak_aix0.1", f"quest16softnative_match-{POLICY}"),
               (POLICY, f"topk_match-{POLICY}"),
               ("quest16softnative_match-questp2048", "questp2048"),
               ("estp_delta0.84_age32_cal0.05w32a8peak_aix0.1_p0.2", "quest16softnative_match-questp2048")]


def sign_test(k, n):
    """Two-sided exact sign test: k of n nonzero differences positive."""
    from math import comb
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0


def ruler_detail(path=RES / "ruler" / "8b_32768_main.json", n_boot=10000):
    """RULER at 32k, per sample. paper_ruler_klq: the question-token KL ratio per task (geometric mean over
    samples) and the share of answers identical to the exact model's. paper_ruler_paired: two conditions on
    the same samples, the KL ratio of one to the other (geometric mean, bootstrap interval with samples
    resampled within each task, two-sided sign test) and the difference of the task average.
    paper_ruler_steps: question and decode lengths, the share of decode steps inside the 32-step warm-up,
    the policy's refresh rate (head-steps, sparse layers, after the first step), the share of blocks its sets
    skip on steps without a refresh (skip_estimate) and the bytes those sets would cost at the mean refresh
    rate of the six 32k documents (paper_policy_sets)."""
    runs = json.loads(Path(path).read_text())["runs"]
    tasks = [t for t in RULER_TASKS if t in {r["task"] for r in runs}]
    recs = []
    for r in runs:
        base = {"task": r["task"], "sample": r["sample"]}
        recs.append({**base, "condition": "sdpa", "score": r["sdpa"]["score"], "kl_q": 1.0,
                     "same": r["sdpa"]["answer_same"], "steps": r["sdpa"]["steps"], "refresh_rate": np.nan})
        for name, c in r["conditions"].items():
            recs.append({**base, "condition": name, "score": c["score"], "kl_q": c["kl_q_x_noise"],
                         "same": c["answer_same"], "steps": c["steps"], "refresh_rate": c["refresh_rate"],
                         "bytes_blocks": c["bytes_blocks"], "bytes_meta": c["bytes_meta"]})
    d = pd.DataFrame(recs)
    d["log_kl"] = np.log(d.kl_q.clip(lower=1e-6))
    label = {k: v.strip() for k, v in RULER_ROWS.items()}
    g = d.groupby(["condition", "task"])
    t = np.exp(g.log_kl.mean()).unstack()[tasks]
    t["all"] = np.exp(d.groupby("condition").log_kl.mean())
    same = g.same.mean().unstack()[tasks].add_prefix("same_")
    same["same_all"] = d.groupby("condition").same.mean()
    t = t.join(same)
    save_table(t.loc[[c for c in RULER_ROWS if c in t.index]].rename(index=label), "paper_ruler_klq")

    rng = np.random.default_rng(0)
    rows = []
    for a, b in RULER_PAIRS:
        x, y = (d[d.condition == c].set_index(["task", "sample"]).sort_index() for c in (a, b))
        lr, ds = x.log_kl - y.log_kl, x.score - y.score
        per = {tk: (lr.loc[tk].to_numpy(), ds.loc[tk].to_numpy()) for tk in tasks}
        bl, bd = [], []
        for _ in range(n_boot):
            ix = {tk: rng.integers(0, len(v[0]), len(v[0])) for tk, v in per.items()}
            bl.append(np.concatenate([per[tk][0][ix[tk]] for tk in tasks]).mean())
            bd.append(np.mean([per[tk][1][ix[tk]].mean() for tk in tasks]))
        k, n = int((lr > 0).sum()), int((lr != 0).sum())
        rows.append({"method": label[a], "against": label[b], "kl_ratio": float(np.exp(lr.mean())),
                     "kl_ci_lo": float(np.exp(np.quantile(bl, 0.025))),
                     "kl_ci_hi": float(np.exp(np.quantile(bl, 0.975))),
                     **{f"kl_ratio_{tk}": float(np.exp(per[tk][0].mean())) for tk in tasks},
                     "further_share": k / len(lr), "sign_p": sign_test(k, n),
                     "score_diff": float(np.mean([per[tk][1].mean() for tk in tasks])) * 100,
                     "score_ci_lo": float(np.quantile(bd, 0.025)) * 100,
                     "score_ci_hi": float(np.quantile(bd, 0.975)) * 100})
    save_table(pd.DataFrame(rows).set_index("method"), "paper_ruler_paired")

    q = pd.DataFrame([{"task": r["task"], "question": r["question_tokens"]} for r in runs])
    pol = d[d.condition == POLICY].copy()
    pol["skip"] = skip_estimate(pol.bytes_blocks, pol.refresh_rate)
    r_docs = policy_sets_docs().refresh_rate.mean()
    rows = []
    for tk in tasks:
        qq, p = q[q.task == tk].question, pol[pol.task == tk]
        rows.append({"task": tk, "question_min": qq.min(), "question_median": qq.median(), "question_max": qq.max(),
                     "steps_min": p.steps.min(), "steps_median": p.steps.median(), "steps_max": p.steps.max(),
                     "warmup_share": np.minimum(p.steps, 32).sum() / p.steps.sum(),
                     "refresh_rate": p.refresh_rate.mean(), "bytes_total": (p.bytes_blocks + p.bytes_meta).mean(),
                     "skip_nonrefresh_est": p.skip.mean(),
                     "bytes_at_docs_refresh_rate": (block_bytes(p.skip, r_docs) + p.bytes_meta).mean()})
    return save_table(pd.DataFrame(rows).set_index("task"), "paper_ruler_steps")


def skip_estimate(bytes_blocks, refresh_rate, layers=32):
    """Share of blocks the policy skips on head-steps without a refresh, from its block bytes and refresh rate
    (layer 0 dense, a refreshing head reads every block). On the document runs, where run_reuse records it,
    the estimate is 0.002-0.005 below the recorded share (paper_policy_sets)."""
    return 1 - ((layers * bytes_blocks - 1) / (layers - 1) - refresh_rate) / (1 - refresh_rate)


def block_bytes(skip, refresh_rate, layers=32):
    """The inverse: block bytes of sets that skip a share `skip` when heads refresh at `refresh_rate`."""
    return (1 + (layers - 1) * (refresh_rate + (1 - refresh_rate) * (1 - skip))) / layers


def policy_sets_docs():
    """The policy on the six 32k documents of the equal-bytes comparison (stage-4 runs s4p and s4b): bytes,
    refresh rate and the share of blocks skipped on head-steps without a refresh, as recorded by run_reuse
    and as estimated from bytes and refresh rate (the estimate used for RULER)."""
    rows = []
    for doc, run in (("prose", "s4p"), ("code", "s4p"), ("dialog", "s4p"),
                     ("prose2", "s4b"), ("code2", "s4b"), ("dialog2", "s4b")):
        c = json.loads((REUSE / f"8b_{doc}_32768.{run}.summary.json").read_text())["conditions"][POLICY]
        rows.append({"doc": doc, "bytes_total": c["bytes_total"], "refresh_rate": c["refresh_rate"],
                     "skip_nonrefresh": c["skip_nonrefresh"],
                     "skip_nonrefresh_est": skip_estimate(c["bytes_blocks"], c["refresh_rate"])})
    return save_table(pd.DataFrame(rows).set_index("doc"), "paper_policy_sets")


def fig_dense():
    """Appendix: the dense decode kernel against SDPA and FlashAttention-2 on one Llama-3.1-8B layer, as a
    share of the RTX 4090's 1008 GB/s (stage2_bench_summary), with the measured streaming-read ceiling
    (results/bench/decode_dense.json, written to paper_kernel_ceiling)."""
    b = pd.read_csv(TAB / "stage2_bench_summary.csv")
    ceiling = json.loads((RES / "bench" / "decode_dense.json").read_text())["ceiling"]
    save_table(pd.DataFrame({"GBs": ceiling, "share_of_1008": {k: v / 1008 for k, v in ceiling.items()}}),
               "paper_kernel_ceiling")
    series = [("sdpa %peak", "SDPA (default)", COLORS["sdpa"]), ("sdpa_cudnn %peak", "SDPA, cuDNN", COLORS["quest_kv"]),
              ("fa2 %peak", "FlashAttention-2", COLORS["questp"]), ("kvskip %peak", "ours (dense)", COLORS["policy"])]
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    w = 0.2
    for i, (col, label, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, b[col], width=w * 0.92, color=color, label=label, linewidth=0)
    ax.axhline(100 * ceiling["read_GBs"] / 1008, color="#52514e", lw=0.6, ls="--")
    ax.text(-0.45, 100 * ceiling["read_GBs"] / 1008 + 0.8, "streaming-read ceiling", ha="left", va="bottom", fontsize=6.5)
    labels = [f"{c // 1024}k\nbs={n}" for c, n in zip(b.ctx, b.bs)]
    ax.set_xticks(x, labels, fontsize=6.5)
    ax.set_ylim(0, 102)
    ax.set_ylabel("% of 1008 GB/s")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.5, borderaxespad=0.1)
    ax.grid(axis="x", visible=False)
    save_fig(fig, "paper_fig_dense")


RULER_SHORT = {"exact (fp32)": "exact (fp32)", "stock bf16 SDPA": "stock bf16 SDPA",
               "refresh + calibration (ours)": "ours", "  + head (2,5) latched to the index": "\\quad + (2,\\,5) latched",
               "Quest per KV head, our bytes": "Quest per KV head", "per-step oracle, our bytes": "per-step oracle",
               "Quest as published, 2048 tokens": "Quest as published", "Quest per KV head, Quest's bytes": "Quest per KV head",
               "per-step oracle, Quest's bytes": "per-step oracle", "refresh + calibration, top-p 0.2": "ours, top-$p$ 0.2"}


def ruler_latex(t: pd.DataFrame) -> str:
    """The paper's RULER table rows, printed from paper_ruler so that no number is copied by hand."""
    lines = []
    for label, r in t.iterrows():
        if label in ("refresh + calibration (ours)", "Quest as published, 2048 tokens"):
            lines.append("\\midrule")
        tasks = " & ".join(f"{r[c]:.0f}" for c in RULER_TASKS if c in r)
        v = r["kl_q_gmean"]
        kl = "--" if np.isnan(v) else f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"
        lines.append(f"{RULER_SHORT[label]} & {100 * r['bytes']:.0f}\\% & {tasks} & {r['average']:.1f} & {kl} \\\\")
    return "\n".join(lines)


def fig_budget():
    """Two operating points, from stage 5: KL against K+V bytes on the six documents (stage5_budget_points,
    geometric means), at column width. Quest's passkey task is an appendix table (stage5_passkey)."""
    p = pd.read_csv(TAB / "stage5_budget_points.csv")
    p.columns = ["method", "point"] + list(p.columns[2:])
    fig, a = plt.subplots(figsize=(COL, 2.15))
    series = [("refresh", "policy", "o", "-", "refresh + calibration (ours)"),
              ("questp", "questp", "s", "-", "Quest as published"),
              ("quest_s4best", "quest_kv", "D", "--", "Quest, pages per KV head"),
              ("topk", "oracle", "v", ":", "per-step oracle (needs all of K)"),
              ("static", "static", "^", ":", "static (first step's blocks)")]
    for method, key, marker, ls, label in series:
        d = p[p.method == method].sort_values("bytes_total")
        a.plot(d.bytes_total, d.kl_x_noise, marker=marker, ms=3.2, lw=1.0, ls=ls, color=COLORS[key], label=label)
    v = p[p.method == "vskip1"]
    a.plot(v.bytes_total, v.kl_x_noise, marker="P", ms=4, ls="none", color="#52514e", label="single-pass V-skip")
    a.axhspan(0.84, 1.43, color="#e2e2df", lw=0, zorder=0)
    a.set_yscale("log")
    a.set_xlim(0.1, 0.82)
    a.set_xlabel("K+V bytes, share of dense")
    a.set_ylabel("KL ratio to bf16 noise")
    a.grid(axis="x", visible=False)
    away = {"p0.4": (3, 2, "left"), "p0.6": (3, 2, "left"), "p0.9": (5, 0, "left"), "p0.99": (3, 2, "left")}
    for _, r in p[p.method == "refresh"].iterrows():
        dx, dy, ha = away.get(r.point, (-3, -8, "right"))
        a.annotate(r.point.replace("p", "$p$="), (r.bytes_total, r.kl_x_noise), xytext=(dx, dy), ha=ha,
                   textcoords="offset points", fontsize=5.5, color=COLORS["policy"])
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.3, handlelength=1.8,
             columnspacing=1.0, borderaxespad=0.2)
    save_fig(fig, "paper_fig4_budget")


def main():
    t = ablation_topp()
    print(t.to_string(float_format=lambda v: f"{v:.3g}"))
    reuse_lag()
    j = fig_jumps()
    print(f"\nfig1: max newly lost {j.kv_heads_newly_lost.max():.0f} of {j.sparse_kv_heads.iloc[0]}; "
          f"mean KL sdpa {j.sdpa.mean():.2e} / timer {j.timer_N32.mean():.2e} / policy {j.policy.mean():.2e} "
          f"(s4q sdpa {j.sdpa_s4q.mean():.2e})")
    fig_reads()
    b = fig_blind()
    print(f"fig3: (2, 5) real miss > 0.4 on {(b.alone_real_miss > 0.4).sum()} steps alone, "
          f"{(b.calibrated_real_miss > 0.4).sum()} calibrated; max c {b.calibrated_c.max():.2f}")
    ci_tables()
    lab_kl()
    timer_equal()
    equal_tests()
    trigger_fail_heads()
    carry_list_reference()
    fig_dense()
    fig_budget()
    if (RES / "ruler" / "8b_32768_main.json").exists():
        r = ruler_tables()
        ruler_detail()
        print("\nTable ruler:\n" + ruler_latex(r))


if __name__ == "__main__":
    main()
