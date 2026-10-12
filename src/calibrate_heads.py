"""Stage 3 (4.2): the always-exact heads of a model from one diagnostic run, by script.

Input: results/reuse/<model>_*.<grid>.heads.parquet of a run_reuse.py run that includes
the timer:32 condition (its stale set lives 32 steps, so the losses a trigger has to catch
are observable). A head's loss is "invisible" when its stale set covers < 0.95 of the exact
mass while the trigger's estimate says < 0.02 (miss_p, the pre-pass estimate, if recorded,
else miss_d). Heads are ranked by the rate of invisible losses; the list is the smallest
top set that holds `--share` of all invisible losses, capped at `--max-heads`. For each
head the token / bigram index coverage (stage 3 diagnostics, if recorded) says whether an
index could replace "always exact".

Prints the list in run_reuse.py's dh=L.H,... syntax; writes results/tables/stage3_calib_<model>.{csv,md}.
Stage 2 picked the Llama-3.1-8B list by hand from the same quantities (8 heads, 44% of the
invisible losses); `python src/calibrate_heads.py --model 8b --grid diag` reproduces it.
"""

import argparse

import pandas as pd

from analyze_stage2 import save_table

import common


def calibrate(model: str, grid: str, share: float, max_heads: int, est: str | None = None):
    files = sorted((common.ROOT / "results" / "reuse").glob(f"{model}_*.{grid}.heads.parquet"))
    if not files:
        raise SystemExit(f"no {model} *.{grid}.heads.parquet")
    frames = []
    for f in files:
        h = pd.read_parquet(f)
        g = h[(h.condition == "timer_N32") & (h.layer > 0) & (h.refresh == 0)].copy()
        g["doc"] = f.name.split(".")[0]
        frames.append(g)
    g = pd.concat(frames)
    est = est or ("miss_p" if "miss_p" in g and g["miss_p"].abs().sum() > 0 else "miss_d")
    g["big"] = g.covered_stale_min < 0.95
    g["invisible"] = g.big & (g[est] < 0.02)
    agg = {"big_loss_rate": ("big", "mean"), "invisible_rate": ("invisible", "mean")}
    for col in ("idx_cov", "idx_kept", "idx2_cov", "idx2_kept", "m_after", "m_same"):
        if col in g:
            agg[col] = (col, "mean")
    per = g.groupby(["layer", "kv_head"]).agg(**agg)
    per["docs_with_invisible"] = g[g.invisible].groupby(["layer", "kv_head"]).doc.nunique()
    per = per.fillna(0).sort_values("invisible_rate", ascending=False)
    per["cum_share"] = per["invisible_rate"].cumsum() / per["invisible_rate"].sum()
    n = min(max_heads, int((per["cum_share"] < share).sum()) + 1)
    chosen = per.head(n)
    save_table(per.head(max(16, n)), f"stage3_calib_{model}", floatfmt=".3f")
    spec = "dh=" + ",".join(f"{l}.{h}" for l, h in chosen.index)
    print(f"{model}: {len(files)} runs, estimate {est}; {n} heads hold {chosen['cum_share'].iloc[-1]:.0%} of the "
          f"invisible losses\n{spec}")
    return chosen, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b")
    ap.add_argument("--grid", default="diag")
    ap.add_argument("--share", type=float, default=0.44)
    ap.add_argument("--max-heads", type=int, default=8)
    ap.add_argument("--est", default=None, help="miss_p or miss_d (default: miss_p if recorded)")
    a = ap.parse_args()
    calibrate(a.model, a.grid, a.share, a.max_heads, a.est)


if __name__ == "__main__":
    main()
