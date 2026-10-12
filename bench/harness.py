"""Timing and correctness helpers shared by the decode benchmarks and tests.

Timing: CUDA events around each call, after an L2 sweep, median of `iters`.
The 4090 has a 72 MB L2: one layer of KV at bs=1/8k (32 MB) would otherwise
be served from cache and report more than DRAM bandwidth. The sweep *reads*
a 256 MB buffer, so L2 is left full of clean lines (a memset would leave dirty
lines whose write-back steals bandwidth from the timed kernel). The sweep also
keeps the stream busy while the CPU enqueues the timed call, so CPU launch
cost is excluded and only GPU time is measured.
"""

import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PEAK_BW = 1008e9          # RTX 4090: 21 Gbit/s x 384 bit / 8
LLAMA8B = dict(Hq=32, Hkv=8, D=128)


class L2Sweep:
    def __init__(self, mb: int = 256):
        self.buf = torch.ones(mb * 2**20 // 4, dtype=torch.float32, device="cuda")
        self.out = torch.empty((), dtype=torch.float32, device="cuda")

    def __call__(self):
        torch.sum(self.buf, dim=0, out=self.out)


_SWEEP = None


def time_cuda(fn, iters: int = 50, warmup: int = 5, flush: bool = True) -> dict:
    """Median / p10 / p90 GPU time of fn() in microseconds."""
    global _SWEEP
    if flush and _SWEEP is None:
        _SWEEP = L2Sweep()
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush:
            _SWEEP()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    q = lambda p: t[min(len(t) - 1, int(p * len(t)))]
    return {"us": statistics.median(t), "us_p10": q(0.1), "us_p90": q(0.9)}


def kv_bytes(B: int, T: int, Hkv: int, D: int, elem: int = 2) -> int:
    """Bytes of K + V one dense decode step must read for one layer."""
    return 2 * B * Hkv * T * D * elem


def bandwidth(nbytes: int, us: float) -> tuple[float, float]:
    """(GB/s, % of the 4090's 1008 GB/s)."""
    bw = nbytes / (us * 1e-6)
    return bw / 1e9, 100.0 * bw / PEAK_BW


def read_ceiling(gb: float = 2.0) -> dict:
    """Achievable DRAM bandwidth (GB/s) on this card over a buffer far larger than L2:
    a streaming 16-byte-load read kernel (best over grid sizes), torch.sum of bf16
    and a device-to-device copy (read + write bytes)."""
    n = int(gb * 2**30) // 2
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    out = torch.empty((), dtype=torch.float32, device="cuda")
    r = time_cuda(lambda: torch.sum(x, dim=0, dtype=torch.float32, out=out), iters=20, flush=False)
    y = torch.empty_like(x)
    c = time_cuda(lambda: y.copy_(x), iters=20, flush=False)
    del y
    res = {"torch_sum_GBs": 2 * n / (r["us"] * 1e-6) / 1e9, "copy_GBs": 4 * n / (c["us"] * 1e-6) / 1e9}
    try:
        from kernels import kvskip
        o = torch.zeros(1, dtype=torch.int32, device="cuda")
        best = 0.0
        for blocks in (128 * 4, 128 * 8, 128 * 16, 128 * 32):
            t = time_cuda(lambda: kvskip.ext().read_bw(x, o, blocks), iters=20, flush=False)
            best = max(best, 2 * n / (t["us"] * 1e-6) / 1e9)
        res["read_GBs"] = best
    except Exception as e:  # extension not built
        res["read_GBs"] = res["torch_sum_GBs"]
        res["read_note"] = str(e)[:100]
    del x
    return res


# ---------------------------------------------------------------- correctness

def reference_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                     keep: torch.Tensor | None = None, block: int = 64) -> torch.Tensor:
    """fp64 decode attention. q [B, Hq, D], k/v [B, Hkv, T, D] -> [B, Hq, D] fp64.

    keep: optional bool [B, Hkv, nb] block mask; dropped blocks are excluded
    before the softmax (what a block-sparse kernel that never loads them computes).
    """
    B, Hq, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    if B > 1:      # one batch element at a time keeps the fp64 temporaries small
        return torch.cat([reference_decode(q[i:i + 1], k[i:i + 1], v[i:i + 1], scale,
                                           None if keep is None else keep[i:i + 1], block) for i in range(B)])
    G = Hq // Hkv
    q64 = q.double().view(B, Hkv, G, D)
    s = torch.einsum("bkgd,bktd->bkgt", q64, k.double()) * scale
    if keep is not None:
        tok = keep.repeat_interleave(block, dim=-1)[..., :T]          # [B, Hkv, T]
        s = s.masked_fill(~tok[:, :, None, :], float("-inf"))
    p = torch.softmax(s, -1)
    return torch.einsum("bkgt,bktd->bkgd", p, v.double()).reshape(B, Hq, D)


def reference_block_stats(q, k, scale, block: int = 64):
    """Per-block max score and sum exp(s - block max): [B, Hq, nb] fp64 each (natural log units)."""
    B, Hq, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    G = Hq // Hkv
    s = torch.einsum("bkgd,bktd->bkgt", q.double().view(B, Hkv, G, D), k.double()) * scale
    nb = math.ceil(T / block)
    s = torch.nn.functional.pad(s, (0, nb * block - T), value=float("-inf")).view(B, Hkv, G, nb, block)
    M = s.amax(-1)
    S = torch.exp(s - M[..., None]).sum(-1)
    return M.reshape(B, Hq, nb), S.reshape(B, Hq, nb)


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.double() - ref).norm() / ref.norm()).item()
