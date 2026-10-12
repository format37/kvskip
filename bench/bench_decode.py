"""B1/B2/B3: decode attention timing, one Llama-3.1-8B layer (Hq=32, Hkv=8, D=128, bf16).

Backends
  sdpa          torch SDPA, q_len=1, enable_gqa=True (HF's decode path); default dispatch
  sdpa_<be>     the same with one backend forced (flash / efficient / cudnn / math), where it runs
  fa2           flash_attn_with_kvcache, its native [B, T, Hkv, D] cache, default split heuristic
  kvskip        our dense split-K kernel, HF cache layout [B, Hkv, T, D], split heuristic
  kvskip_best   kvskip with the best split count of a sweep
  kvskip_bshd   kvskip on FA2's memory layout (same bytes as fa2)
Grid: bs {1, 8, 32} x ctx {8k, 32k, 64k, 128k}. Timing: harness.time_cuda (L2 sweep,
CUDA events, median of 50). Bandwidth counts the K + V bytes of one dense step.

    .venv/bin/python bench/bench_decode.py [--backends sdpa fa2 kvskip] [--tag name]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import LLAMA8B, ROOT, bandwidth, kv_bytes, read_ceiling, time_cuda  # noqa: E402

sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "bench"
BS = (1, 8, 32)
CTX = (8192, 32768, 65536, 131072)
SDPA_BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                 "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}


def rand_kv(B, T, Hkv, D, layout):
    shape = (B, Hkv, T, D) if layout == "bhsd" else (B, T, Hkv, D)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return k, v


def try_time(fn, **kw):
    try:
        fn()
        torch.cuda.synchronize()
        return time_cuda(fn, **kw)
    except (RuntimeError, torch.OutOfMemoryError) as e:
        torch.cuda.synchronize()
        return {"error": str(e).split("\n")[0][:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["sdpa", "sdpa_all", "fa2", "kvskip"])
    ap.add_argument("--bs", nargs="+", type=int, default=list(BS))
    ap.add_argument("--ctx", nargs="+", type=int, default=list(CTX))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--splits", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 48, 64])
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--tag", default="dense")
    args = ap.parse_args()
    Hq, Hkv, D = LLAMA8B["Hq"], LLAMA8B["Hkv"], LLAMA8B["D"]
    scale = D ** -0.5
    want = set(args.backends)
    if "kvskip" in want:
        from kernels import kvskip
        kvskip.ext()
    if "fa2" in want:
        from flash_attn import flash_attn_with_kvcache

    warm = torch.randn(2**28, device="cuda")
    for _ in range(50):                      # ~0.5 s of load so clocks are up before measuring
        warm.mul_(1.0)
    del warm
    ceiling = read_ceiling()
    print(f"[bench] DRAM ceiling on this card: streaming read {ceiling['read_GBs']:.0f} GB/s, "
          f"torch.sum {ceiling['torch_sum_GBs']:.0f} GB/s, copy {ceiling['copy_GBs']:.0f} GB/s (spec 1008)", flush=True)
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"decode_{args.tag}"

    def rec(backend, B, T, r, **extra):
        nbytes = kv_bytes(B, T, Hkv, D)
        row = {"backend": backend, "bs": B, "ctx": T, **extra}
        if "error" in r:
            row["error"] = r["error"]
            print(f"  {backend:14s} bs={B:2d} ctx={T:6d}  n/a ({r['error'][:60]})", flush=True)
        else:
            gbs, pct = bandwidth(nbytes, r["us"])
            row.update(us=r["us"], us_p10=r["us_p10"], us_p90=r["us_p90"], GBs=gbs, pct_peak=pct)
            print(f"  {backend:14s} bs={B:2d} ctx={T:6d}  {r['us']:9.1f} us  {gbs:6.0f} GB/s  {pct:5.1f}%"
                  + (f"  {extra}" if extra else ""), flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)       # written as we go

    for T in args.ctx:
        for B in args.bs:
            q = torch.randn(B, Hq, D, device="cuda", dtype=torch.bfloat16)
            k, v = rand_kv(B, T, Hkv, D, "bhsd")
            if "sdpa" in want:
                q4 = q[:, :, None]
                rec("sdpa", B, T, try_time(lambda: F.scaled_dot_product_attention(
                    q4, k, v, scale=scale, enable_gqa=True), iters=args.iters))
            if "sdpa_all" in want:
                for name, be in SDPA_BACKENDS.items():
                    if name == "math" and B * T > 8 * 32768:
                        rec(f"sdpa_{name}", B, T, {"error": "skipped (repeats K/V 4x for GQA)"})
                        continue

                    def f(be=be):
                        with sdpa_kernel([be]):
                            return F.scaled_dot_product_attention(q[:, :, None], k, v, scale=scale, enable_gqa=True)
                    rec(f"sdpa_{name}", B, T, try_time(f, iters=args.iters))
            if "kvskip" in want:
                auto = kvskip.num_splits_for(B * Hkv, math.ceil(T / 64))
                rec("kvskip", B, T, try_time(lambda: kvskip.decode(q, k, v, scale, stages=args.stages),
                                             iters=args.iters), splits=auto)
                best = None
                for s in sorted(set(args.splits + [auto])):
                    if s > 1 and math.ceil(math.ceil(T / 64) / s) < 2:
                        continue
                    r = try_time(lambda: kvskip.decode(q, k, v, scale, num_splits=s, stages=args.stages),
                                 iters=max(10, args.iters // 2))
                    if "us" in r and (best is None or r["us"] < best[1]["us"]):
                        best = (s, r)
                if best:
                    r = time_cuda(lambda: kvskip.decode(q, k, v, scale, num_splits=best[0], stages=args.stages),
                                  iters=args.iters)
                    rec("kvskip_best", B, T, r, splits=best[0])
            del k, v
            torch.cuda.empty_cache()
            if "fa2" in want or "kvskip" in want:
                kc, vc = rand_kv(B, T, Hkv, D, "bshd")
                if "fa2" in want:
                    q4 = q[:, None]
                    rec("fa2", B, T, try_time(lambda: flash_attn_with_kvcache(q4, kc, vc, softmax_scale=scale),
                                              iters=args.iters))
                if "kvskip" in want:
                    kt, vt = kc.transpose(1, 2), vc.transpose(1, 2)
                    rec("kvskip_bshd", B, T, try_time(lambda: kvskip.decode(q, kt, vt, scale, stages=args.stages),
                                                      iters=args.iters))
                    del kt, vt
                del kc, vc
                torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)
    meta = {"ceiling": ceiling, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "args": vars(args), "time": time.strftime("%Y-%m-%d %H:%M")}
    Path(f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(f"[bench] wrote {stem}.csv")


if __name__ == "__main__":
    main()
