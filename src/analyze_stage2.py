"""Stage 2 analysis: A1 (selection reuse, triggers, single-pass V-skip rules),
A2 (mask locality, block size, cheap block rankings), A3 (perplexity) and the
kernel benchmarks -> results/tables/stage2_*.{csv,md}, results/figures/stage2_*.png.
Robust to partial results.
"""

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


def save_table(df: pd.DataFrame, name: str, floatfmt=".3g"):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv")
    (TAB / f"{name}.md").write_text(df.to_markdown(floatfmt=floatfmt))
    return df


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=110)
    plt.close(fig)


def auroc(score, label) -> float:
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------- A1

def family(name: str) -> str:
    for f in ("fresh", "timer", "oracle", "drift", "estc", "estd", "vtau", "vsink", "vprev2", "vprev"):
        if name.startswith(f):
            return f
    return "other"


def load_reuse() -> pd.DataFrame:
    rows = []
    for f in sorted((RES / "reuse").glob("8b_*.summary.json")):
        s = json.loads(f.read_text())
        grid = f.name.split(".")[1] if f.name.count(".") > 2 else "main"
        for name, c in s["conditions"].items():
            rows.append({"grid": grid, "set": "heldout" if grid == "heldout" else "main",
                         "genre": s["genre"], "ctx": s["ctx"], "condition": name,
                         "family": family(name), "noise_kl": s["noise_kl_mean"], **c})
    df = pd.DataFrame(rows)
    # the same condition can appear in several grids (deterministic re-runs): keep the first
    return df.drop_duplicates(["genre", "ctx", "condition"])


def a1_tables(df: pd.DataFrame, held: bool = False):
    df = df.copy()
    df["kv_skip"] = (df["k_skip"] + df["v_skip"]) / 2      # share of K+V bytes not read
    long = df.set_index(["condition", "ctx", "genre"])[
        ["kl_x_noise", "kl_p95_x_noise", "top1_agree", "k_skip", "v_skip", "kv_skip", "refresh_rate"]
    ].sort_index()
    save_table(long, "stage2_a1_heldout_by_doc" if held else "stage2_a1_reuse_by_doc")
    g = df.groupby(["condition", "ctx"])
    summ = pd.DataFrame({
        "family": g["family"].first(),
        "KL_x_noise_mean": g["kl_x_noise"].mean(),
        "KL_x_noise_max": g["kl_x_noise"].max(),
        "p95_x_noise_max": g["kl_p95_x_noise"].max(),
        "top1": g["top1_agree"].mean(),
        "K_skip": g["k_skip"].mean(),
        "V_skip": g["v_skip"].mean(),
        "KV_bytes_skip": g["kv_skip"].mean(),
        "refresh_rate": g["refresh_rate"].mean(),
    }).reset_index()
    order = {f: i for i, f in enumerate(("fresh", "timer", "oracle", "drift", "estc", "estd",
                                        "vtau", "vprev", "vprev2", "vsink"))}
    summ["o"] = summ["family"].map(order).fillna(99)
    summ["n"] = summ["condition"].str.extract(r"timer_N(\d+)")[0].astype(float).fillna(0)
    summ = summ.sort_values(["ctx", "o", "n", "condition"]).drop(columns=["o", "n"]).set_index(["ctx", "condition"])
    save_table(summ, "stage2_a1_heldout" if held else "stage2_a1_reuse")
    return summ


def a1_tradeoff(summ: pd.DataFrame):
    colors = {"fresh": "k", "timer": "tab:red", "oracle": "tab:green", "drift": "tab:orange",
              "estc": "tab:purple", "estd": "tab:blue", "vtau": "tab:gray", "vprev": "tab:brown",
              "vprev2": "tab:olive", "vsink": "tab:cyan"}
    labels = {"fresh": "fresh top-p 0.99 (V-skip, 2 passes)", "timer": "refresh every N steps",
              "oracle": "refresh when coverage < 0.98 (oracle)", "drift": "refresh on query drift",
              "estc": "refresh on centroid estimate", "estd": "refresh on lse+centroid-drift estimate",
              "vtau": "V-skip W<1e-4 (oracle)", "vprev": "V-skip 1 pass, prev-step lse",
              "vprev2": "V-skip 1 pass, prev lse + 2 nats", "vsink": "V-skip 1 pass, sink bound"}
    ctxs = sorted(summ.index.get_level_values(0).unique())
    fig, axes = plt.subplots(1, len(ctxs), figsize=(8 * len(ctxs), 6), constrained_layout=True, squeeze=False)
    for ax, ctx in zip(axes[0], ctxs):
        s = summ.loc[ctx]
        seen = set()
        for name, r in s.iterrows():
            fam = r["family"]
            dh = "_dh" in name
            ax.errorbar(100 * r["KV_bytes_skip"], r["KL_x_noise_mean"],
                        yerr=[[0], [max(0.0, r["KL_x_noise_max"] - r["KL_x_noise_mean"])]],
                        fmt="s" if dh else "o", color=colors.get(fam, "k"), capsize=3, ms=6,
                        mfc="white" if dh else colors.get(fam, "k"),
                        label=None if fam in seen else labels.get(fam, fam))
            seen.add(fam)
            short = (name.replace("_tau0.0001", "").replace("_age32", "").replace("timer_", "")
                     .replace("oracle_theta", "θ").replace("_delta", " δ").replace("estd", "est")
                     .replace("estc", "c"))
            if fam in ("timer", "estd", "oracle") or dh:
                ax.annotate(short, (100 * r["KV_bytes_skip"], r["KL_x_noise_mean"]), fontsize=7,
                            xytext=(4, 3), textcoords="offset points")
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.axhspan(0.5, 1.5, color="green", alpha=0.07)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 200)
        ax.set_xlabel("K+V bytes not read, % of dense (mean over steps; layer 0 dense)")
        ax.set_ylabel("KL / KL(stock bf16 SDPA): mean of 3 genres, bar to worst")
        ax.set_title(f"Llama-3.1-8B, {ctx // 1024}k context, 128 decode steps")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, "stage2_a1_tradeoff")


def a1_jumps():
    """Per-step KL of the timer policy vs the trigger, with the coverage dips."""
    files = [RES / "reuse" / f"8b_{g}_32768.steps.parquet" for g in ("code", "prose")]
    files = [f for f in files if f.exists()]
    if not files:
        return
    fig, axes = plt.subplots(2, len(files), figsize=(8 * len(files), 7), constrained_layout=True,
                             sharex=True, squeeze=False)
    for c, f in enumerate(files):
        s = pd.read_parquet(f)
        ax, ax2 = axes[0, c], axes[1, c]
        for cond, col in (("stock_sdpa_bf16", "gray"), ("fresh", "k"), ("timer_N16", "tab:red"),
                          ("oracle_theta0.98", "tab:green")):
            t = s[s.condition == cond]
            if len(t):
                ax.plot(t.step, t.kl, color=col, lw=1, label=cond)
                if cond.startswith(("timer", "oracle")):
                    ax2.plot(t.step, t.covered_min, color=col, lw=1, label=cond)
        t = s[s.condition == "timer_N16"]
        for st in t.step[(t.step % 16 == 0)]:
            ax.axvline(st, color="tab:red", alpha=0.15, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("KL per step (nats)")
        ax.set_title(f"{f.name.split('.')[0]}: per-step KL (red lines: timer refreshes)")
        ax.legend(fontsize=8)
        ax2.set_ylabel("min over layers/KV heads of attended mass")
        ax2.set_xlabel("decode step")
        ax2.set_ylim(0, 1.02)
        ax2.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax2.grid(alpha=0.3)
    save_fig(fig, "stage2_a1_jumps")


def a1_where_and_triggers():
    """Which layers lose coverage under a stale set, and which cheap signal predicts it."""
    rows, auc_rows = [], []
    for f in sorted((RES / "reuse").glob("8b_*.diag.heads.parquet")):
        h = pd.read_parquet(f)
        genre, ctx = f.name.split("_")[1], int(f.name.split("_")[2].split(".")[0])
        g = h[(h.condition == "timer_N32") & (h.layer > 0) & (h.refresh == 0)]
        for lab in (0.95, 0.9):
            y = g.covered_stale_min < lab
            auc_rows.append({"ctx": ctx, "genre": genre, "event": f"stale coverage < {lab}",
                             "event_rate": float(y.mean()),
                             "AUROC query drift": auroc(g.drift, y), "AUROC miss_c (centroid)": auroc(g.miss_c, y),
                             "AUROC miss_d (lse + centroid drift)": auroc(g.miss_d, y),
                             "AUROC age": auroc(g.age, y)})
        per_layer = g.assign(low=g.covered_stale_min < 0.95).groupby("layer").agg(
            low=("low", "mean"), cov=("covered_stale_min", "mean"))
        per_layer["ctx"], per_layer["genre"] = ctx, genre
        rows.append(per_layer.reset_index())
    if not rows:
        return
    auc = pd.DataFrame(auc_rows).set_index(["event", "ctx", "genre"]).sort_index()
    save_table(auc, "stage2_a1_trigger_auroc")
    # heads whose big coverage losses the centroid-drift estimate does not flag
    frames = []
    for f in sorted((RES / "reuse").glob("8b_*.diag.heads.parquet")):
        h = pd.read_parquet(f)
        g = h[(h.condition == "timer_N32") & (h.layer > 0) & (h.refresh == 0)].copy()
        g["big"] = (1 - g.covered_stale_min) > 0.05
        g["unflagged"] = g.big & (g.miss_d < 0.02)
        g["doc"] = f.name.split(".")[0]
        frames.append(g)
    g = pd.concat(frames)
    per = g.groupby(["layer", "kv_head"]).agg(big_miss_rate=("big", "mean"), unflagged_rate=("unflagged", "mean"))
    per["docs_with_unflagged"] = g[g.unflagged].groupby(["layer", "kv_head"]).doc.nunique()
    per = per.fillna(0).sort_values("unflagged_rate", ascending=False)
    per["cum_share_of_unflagged"] = per["unflagged_rate"].cumsum() / per["unflagged_rate"].sum()
    save_table(per.head(16), "stage2_a1_unflagged_heads")
    pl = pd.concat(rows)
    t = pl.pivot_table(index="layer", columns="ctx", values="low", aggfunc="mean")
    t.columns = [f"P(stale cov < 0.95) {c // 1024}k" for c in t.columns]
    t2 = pl.pivot_table(index="layer", columns="ctx", values="cov", aggfunc="mean")
    t2.columns = [f"mean stale cov {c // 1024}k" for c in t2.columns]
    save_table(pd.concat([t, t2], axis=1), "stage2_a1_jump_layers")
    return auc


# ------------------------------------------------------------------- A2

def a2_locality():
    rows, rank_rows, hist = [], [], {}
    for f in sorted((RES / "reuse").glob("8b_*.locality.npz")):
        z = np.load(f)
        genre, ctx = f.name.split("_")[1], int(f.name.split("_")[2].split(".")[0])
        for bs in (16, 32, 64, 128):
            a = {k[len(f"b{bs}_"):]: z[k] for k in z.files if k.startswith(f"b{bs}_")}
            for lay, sl in (("all", slice(None)), ("1-31", slice(1, None))):
                kb, b = a["kept_blocks"][sl].sum(), a["blocks"][sl].sum()
                rows.append({"ctx": ctx, "genre": genre, "block": bs, "layers": lay,
                             "kept_blocks_pct": 100 * kb / b,
                             "kept_tokens_pct": 100 * a["kept_tokens"][sl].sum() / a["tokens"][sl].sum(),
                             "kept_runs": a["kept_runs"][sl].sum(), "skip_runs": a["skip_runs"][sl].sum(),
                             "kept_blocks": kb, "skipped_blocks": a["skipped"][sl].sum(),
                             "kept_in_runs_ge4": a["kept_in_long"][sl].sum(),
                             "skip_in_runs_ge4": a["skip_in_long"][sl].sum()})
            if bs == 64:
                hist[(ctx, genre)] = (a["hist_kept"][1:].sum(0), a["hist_skip"][1:].sum(0))
        c = z["rank_count"]
        for name in ("blockmax", "quest64", "quest32", "quest16", "centroid"):
            rank_rows.append({"ctx": ctx, "genre": genre, "predictor": name,
                              "blocks_ratio_vs_oracle": float(z[f"rank_{name}"][1:].sum() / z["rank_oracle"][1:].sum())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    g = df.groupby(["layers", "ctx", "block"]).sum(numeric_only=True)
    out = pd.DataFrame({
        "kept blocks %": df.groupby(["layers", "ctx", "block"])["kept_blocks_pct"].mean(),
        "kept tokens %": df.groupby(["layers", "ctx", "block"])["kept_tokens_pct"].mean(),
        "mean kept run (blocks)": g["kept_blocks"] / g["kept_runs"],
        "mean skipped run (blocks)": g["skipped_blocks"] / g["skip_runs"],
        "kept blocks in runs >= 4, %": 100 * g["kept_in_runs_ge4"] / g["kept_blocks"],
        "skipped blocks in runs >= 4, %": 100 * g["skip_in_runs_ge4"] / g["skipped_blocks"],
    })
    save_table(out, "stage2_a2_locality")
    rk = pd.DataFrame(rank_rows).pivot_table(index="predictor", columns="ctx", values="blocks_ratio_vs_oracle")
    rk.columns = [f"{c // 1024}k" for c in rk.columns]
    save_table(rk.loc[[p for p in ("blockmax", "centroid", "quest16", "quest32", "quest64") if p in rk.index]],
               "stage2_a2_block_rankings")
    # run-length distribution, block 64, share of blocks in runs of length L
    if hist:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for (ctx, genre), (hk, hs) in sorted(hist.items()):
            L = np.arange(len(hk))
            for ax, hh, what in ((axes[0], hk, "kept"), (axes[1], hs, "skipped")):
                share = hh * L / max(1, (hh * L).sum())
                ax.step(L[1:], np.cumsum(share[1:]), where="post", label=f"{genre} {ctx // 1024}k")
                ax.set_title(f"union top-p 0.99, block 64, layers 1-31: {what} blocks")
                ax.set_xscale("log")
                ax.set_xlabel("run length L (blocks)")
                ax.set_ylabel(f"share of {what} blocks in runs of length <= L")
                ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        save_fig(fig, "stage2_a2_runs")
    return out


# ------------------------------------------------------------------- A3

def a3_ppl():
    rows = []
    for f in sorted((RES / "ppl").glob("8b_*.summary.json")):
        s = json.loads(f.read_text())
        for name, c in s["conditions"].items():
            rows.append({"ctx": s["ctx"], "genre": s["genre"], "condition": name, **c})
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["ppl", "dppl_pct", "dnll", "dnll_se", "top1", "dnll_0_256", "dnll_256_1024", "dnll_1024_2048"]
    t = df.set_index(["ctx", "genre", "condition"])[[c for c in cols if c in df.columns]].sort_index()
    save_table(t, "stage2_a3_ppl", floatfmt=".4g")
    return t


# ------------------------------------------------------------------- bench

def bench():
    files = sorted((RES / "bench").glob("decode_*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f).assign(src=f.stem) for f in files], ignore_index=True)
    df = df[df["us"].notna()].drop_duplicates(["backend", "bs", "ctx"], keep="last")
    t = df.pivot_table(index=["ctx", "bs"], columns="backend", values="us")
    p = df.pivot_table(index=["ctx", "bs"], columns="backend", values="pct_peak")
    save_table(t, "stage2_bench_us", floatfmt=".1f")
    save_table(p, "stage2_bench_pct_peak", floatfmt=".1f")
    cols = [c for c in ("sdpa", "sdpa_cudnn", "fa2", "kvskip", "kvskip_bshd") if c in t.columns]
    summ = pd.DataFrame(index=t.index)
    for c in cols:
        summ[f"{c} us"] = t[c]
    for c in cols:
        summ[f"{c} %peak"] = p[c]
    if "fa2" in t.columns:
        summ["kvskip / fa2 throughput"] = t["fa2"] / t["kvskip"]
    summ["kvskip / sdpa throughput"] = t["sdpa"] / t["kvskip"]
    save_table(summ, "stage2_bench_summary", floatfmt=".3g")
    # figure: % of 1008 GB/s per config
    fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    idx = [f"{c // 1024}k\nbs={b}" for c, b in p.index]
    x = np.arange(len(idx))
    w = 0.8 / len(cols)
    names = {"sdpa": "torch SDPA (default)", "sdpa_cudnn": "torch SDPA, cuDNN backend", "fa2": "FA2 2.8.3",
             "kvskip": "kvskip dense (HF layout)", "kvskip_bshd": "kvskip dense (FA2 layout)"}
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, p[c].values, w, label=names.get(c, c))
    ax.axhline(95.0, color="gray", ls="--", lw=0.8)
    ax.text(len(idx) - 0.5, 95.5, "streaming-read ceiling ~95%", ha="right", fontsize=8, color="gray")
    ax.set_xticks(x, idx, fontsize=8)
    ax.set_ylim(50, 100)
    ax.set_ylabel("% of 1008 GB/s (K+V bytes / time)")
    ax.set_title("Decode attention, one Llama-3.1-8B layer (32 q heads, 8 KV heads, d=128), RTX 4090")
    ax.legend(fontsize=8, ncol=len(cols), loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "stage2_bench_dense")
    return t, p


def sparse_bench():
    f = RES / "bench" / "sparse_real.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    t = df.groupby(["bs", "mode"])[["us", "vs_dense", "k_read", "v_read", "kv_read"]].mean()
    save_table(t, "stage2_bench_sparse_real", floatfmt=".3g")
    f2 = RES / "bench" / "sparse_sweep.csv"
    if f2.exists():
        sw = pd.read_csv(f2).set_index(["ctx", "bs", "target"])[["kept", "us", "vs_dense", "eff_GBs"]]
        save_table(sw, "stage2_bench_sparse_sweep", floatfmt=".3g")
    return t


def template():
    rows = []
    for f in sorted((RES / "template").glob("8b_*.json")):
        r = json.loads(f.read_text())
        m = r["mean_layers_1_31"]
        rows.append({"doc": r["name"], "genre": r["genre"], "union_blocks_0.99_%": 100 * m["union_frac"],
                     "head_blocks_0.99_%": 100 * m["head_frac"], "sink_mass": m["sink"],
                     "service_token_mass": m["svc"], "service_tokens_%_of_prompt": 100 * r["svc_share"]})
    if not rows:
        return
    t = pd.DataFrame(rows).set_index("doc")
    save_table(t, "stage2_a4_template", floatfmt=".3g")
    return t


def e2e():
    rows = []
    for f in sorted((RES / "e2e").glob("8b_*.json")):
        r = json.loads(f.read_text())
        for name, c in r["conditions"].items():
            rows.append({"genre": r["genre"], "ctx": r["ctx"], "condition": name, "kl_x_noise": c["kl_x_noise"],
                         "kl_p95": c["kl_p95"], "top1": c["top1_agree"],
                         "kv_read": c.get("kv_read_frac_sparse_layers"), "refresh_rate": c.get("refresh_rate"),
                         "ms_per_step": r.get("timing", {}).get(name, {}).get("ms_median")})
    if not rows:
        return
    t = pd.DataFrame(rows).set_index(["ctx", "condition", "genre"]).sort_index()
    save_table(t, "stage2_e2e", floatfmt=".3g")
    return t


def main():
    df = load_reuse()
    if len(df):
        summ = a1_tables(df[df["set"] == "main"])
        a1_tradeoff(summ)
        print(summ.round(3).to_string())
        if (df["set"] == "heldout").any():
            print(a1_tables(df[df["set"] == "heldout"], held=True).round(3).to_string())
    a1_jumps()
    auc = a1_where_and_triggers()
    if auc is not None:
        print(auc.round(3).to_string())
    loc = a2_locality()
    if loc is not None:
        print(loc.round(2).to_string())
    t = a3_ppl()
    if t is not None:
        print(t.round(4).to_string())
    bench()
    sb = sparse_bench()
    if sb is not None:
        print(sb.round(3).to_string())
    tp = template()
    if tp is not None:
        print(tp.round(3).to_string())
    ee = e2e()
    if ee is not None:
        print(ee.round(3).to_string())


if __name__ == "__main__":
    main()
