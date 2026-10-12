"""B4: timing of the sparse modes against the dense kernel.

Part 1, real data (data/kv_dumps, Llama-3.1-8B, 32k, layers 0/9/21/31, three genres;
bs=1 per (genre, layer) and bs=8 stacked): per mode the GPU time, the share of
K / V blocks read and the time relative to dense.
  dense           kvskip dense
  masked          stored GQA-union top-p 0.99 set + sink + 2 newest blocks
                  (= a non-refresh step of the refresh policy: K and V of the set only)
  vskip2_kernels  two-pass V-skip, both kernels with a precomputed plan
  vskip2_total    the same plus the PyTorch planner (top-p over block stats, plan)
  vskip1_prev     single-pass V-skip, tau=1e-4, previous-step normaliser (exact lse given)
  vskip1_sink     single-pass V-skip, tau=1e-4, sink-anchored bound
Part 2, synthetic: masked kernel vs dense over kept fractions 1 .. 0.1
(random runs of 1-8 blocks), bs {1, 8}, ctx {32k, 128k}.

    .venv/bin/python bench/bench_sparse.py     # -> results/bench/sparse_real.csv, sparse_sweep.csv
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT, time_cuda  # noqa: E402

sys.path.insert(0, str(ROOT))
from kernels import kvskip  # noqa: E402

DUMPS = ROOT / "data" / "kv_dumps"
OUT = ROOT / "results" / "bench"
ITERS = 50


def load_cases():
    cases = []
    for path in sorted(DUMPS.glob("8b_*_32768.safetensors")):
        with safe_open(str(path), "pt", device="cuda") as f:
            meta = f.metadata()
            T = json.loads(meta["T"])[-1]
            for l in json.loads(meta["layers"]):
                cases.append({"genre": meta["genre"], "layer": l, "T": T, "scale": float(meta["scaling"]),
                              "q": f.get_tensor(f"L{l}.q")[-1], "k": f.get_tensor(f"L{l}.k"),
                              "v": f.get_tensor(f"L{l}.v"),
                              "mask": f.get_tensor(f"L{l}.mask099")[-1].bool()})
    return cases


def modes_for(q, k, v, scale, keep):
    """Return {mode: (callable, k_frac, v_frac)} for one batch."""
    B, Hkv = k.shape[0], k.shape[1]
    nb = math.ceil(k.shape[2] / 64)
    plan = kvskip.make_plan(keep)
    kept = keep.float().mean().item()
    # two-pass: precompute pass-2 inputs once for the kernels-only timing
    r1 = kvskip.decode(q, k, v, scale, stats=True, load_v=False, write_p=True)
    logw = r1["blk_max"] + torch.log(r1["blk_sum"])
    G = q.shape[1] // Hkv
    keep2 = kvskip.select_blocks(logw, G, "topp", 0.99, 2)
    keep2_h = keep2.repeat_interleave(G, dim=1)
    lse_kept = logw.masked_fill(~keep2_h, float("-inf")).logsumexp(-1, keepdim=True)
    sb = torch.exp(r1["blk_max"] - lse_kept).masked_fill(~keep2_h, 0.0)
    sb = sb.view(B, Hkv, G, -1).transpose(2, 3).contiguous()
    plan2 = kvskip.make_plan(keep2)
    kept2 = keep2.float().mean().item()
    lse_exact = kvskip.decode(q, k, v, scale)["lse"]
    v1p = kvskip.vskip_one_pass(q, k, v, scale, tau=1e-4, count=True, lse_prev=lse_exact)["kept_frac"]
    v1s = kvskip.vskip_one_pass(q, k, v, scale, tau=1e-4, count=True)["kept_frac"]

    def vskip2_kernels():
        a = kvskip.decode(q, k, v, scale, stats=True, load_v=False, write_p=True)
        return kvskip.pv(a["p"], v, sb, plan2)

    return {
        "dense": (lambda: kvskip.decode(q, k, v, scale), 1.0, 1.0),
        "masked": (lambda: kvskip.decode(q, k, v, scale, plan=plan), kept, kept),
        "vskip2_kernels": (vskip2_kernels, 1.0, kept2),
        "vskip2_total": (lambda: kvskip.vskip_two_pass(q, k, v, scale, "topp", 0.99, sync=False), 1.0, kept2),
        "vskip1_prev": (lambda: kvskip.vskip_one_pass(q, k, v, scale, tau=1e-4, lse_prev=lse_exact), 1.0, v1p),
        "vskip1_sink": (lambda: kvskip.vskip_one_pass(q, k, v, scale, tau=1e-4), 1.0 + 1.0 / nb, v1s),
    }


def bench_real():
    cases = load_cases()
    rows = []

    def run(tag, q, k, v, scale, keep, extra):
        ms = modes_for(q, k, v, scale, keep)
        base = None
        for name, (fn, kf, vf) in ms.items():
            r = time_cuda(fn, iters=ITERS)
            base = r["us"] if name == "dense" else base
            row = {"case": tag, "mode": name, "us": r["us"], "vs_dense": r["us"] / base,
                   "k_read": kf, "v_read": vf, "kv_read": (kf + vf) / 2, **extra}
            rows.append(row)
            print(f"  {tag:18s} {name:15s} {r['us']:8.1f} us  x{row['vs_dense']:.3f}  "
                  f"K {kf:.2f} V {vf:.2f}", flush=True)

    for c in cases:
        keep = c["mask"][None, :, :math.ceil(c["T"] / 64)].clone()
        keep[..., 0] = True
        keep[..., -2:] = True
        run(f"{c['genre']}/L{c['layer']}", c["q"][None], c["k"][None], c["v"][None], c["scale"], keep,
            {"genre": c["genre"], "layer": c["layer"], "bs": 1})
    # bs=8: the 8 cases of layers 9/21/31 x genres (+ two of layer 0), common length
    sel = [c for c in cases if c["layer"] != 0][:8]
    if len(sel) < 8:
        sel = cases[:8]
    T = min(c["T"] for c in sel)
    q = torch.stack([c["q"] for c in sel])
    k = torch.stack([c["k"][:, :T] for c in sel])
    v = torch.stack([c["v"][:, :T] for c in sel])
    keep = torch.stack([c["mask"][:, :math.ceil(T / 64)] for c in sel]).clone()
    keep[..., 0] = True
    keep[..., -2:] = True
    run("bs8 (layers 9-31)", q, k, v, sel[0]["scale"], keep, {"genre": "mix", "layer": -1, "bs": 8})
    return pd.DataFrame(rows)


def bench_sweep():
    rows = []
    g = torch.Generator(device="cuda").manual_seed(0)
    for T in (32768, 131072):
        for B in (1, 8):
            nb = T // 64
            q = torch.randn(B, 32, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(B, 8, T, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, 8, T, 128, device="cuda", dtype=torch.bfloat16)
            dense = time_cuda(lambda: kvskip.decode(q, k, v, 0.088), iters=ITERS)["us"]
            for frac in (1.0, 0.75, 0.5, 0.25, 0.1):
                # random runs of 1..8 blocks, kept with probability frac
                runs = torch.randint(1, 9, (B, 8, nb), device="cuda", generator=g)
                start = torch.cumsum(runs, -1) - runs
                on = torch.rand(B, 8, nb, device="cuda", generator=g) < frac
                keep = torch.zeros(B, 8, nb, dtype=torch.bool, device="cuda")
                for b in range(B):
                    for h in range(8):
                        st, ln, o = start[b, h], runs[b, h], on[b, h]
                        m = (st < nb) & o
                        for s0, l0 in zip(st[m].tolist(), ln[m].tolist()):
                            keep[b, h, s0:s0 + l0] = True
                keep[..., 0] = True
                plan = kvskip.make_plan(keep)
                kept = keep.float().mean().item()
                us = time_cuda(lambda: kvskip.decode(q, k, v, 0.088, plan=plan), iters=ITERS)["us"]
                rows.append({"ctx": T, "bs": B, "target": frac, "kept": kept, "us": us, "dense_us": dense,
                             "vs_dense": us / dense, "ideal": kept,
                             "eff_GBs": 2 * B * 8 * T * 128 * 2 * kept / (us * 1e-6) / 1e9})
                print(f"  sweep ctx={T} bs={B} kept={kept:.3f}  {us:8.1f} us  x{us / dense:.3f} "
                      f"(ideal x{kept:.3f})", flush=True)
            del q, k, v
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    real = bench_real()
    real.to_csv(OUT / "sparse_real.csv", index=False)
    sweep = bench_sweep()
    sweep.to_csv(OUT / "sparse_sweep.csv", index=False)
    s = real.groupby(["bs", "mode"])[["vs_dense", "k_read", "v_read", "kv_read"]].mean()
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
