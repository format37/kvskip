"""Stage 3: the device-side refresh planner (kernels/kvskip_planner.cu, kvskip_attn mode
"refresh_fused") against the Python planner it replaces ("refresh_pre").

Real data: the stage-2 dumps (src/dump_kv.py: layers 0/9/21/31 of Llama-3.1-8B at 32k,
three genres -> a batch of 3) give the keys/values and the queries of decode steps 0, 1,
16 and 64. The test decodes `steps` steps on them, the query moving linearly between the
dumped ones (so the drift estimate moves, heads trigger, sets age), the cache growing by
one real token per step. Both planners run side by side on the same inputs with separate
state; per step the blocks each KV head reads must agree, and the attention outputs must
match where they do. A borderline estimate or top-p tie (fp32 summation order) may flip a
head, after which the two states differ, so the criterion is an agreement rate.

Stage 4 adds the online calibration of the estimate (KVSKIP.calib): the same runs with the
correction on (log-odds, additive and peak-hold kinds) and no always-exact heads, the warm-up refreshing
every 8 steps for the first 32, and once with heads latched onto the token index when their
correction exceeds 0.001 ("auto"); besides the read sets, the per-head corrections of the two
planners must agree.

Run: .venv/bin/python bench/test_planner.py
"""

import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import kvskip_attn  # noqa: E402
from kvskip_attn import KVSKIP  # noqa: E402

BLOCK = 64
DUMPS = ROOT / "data" / "kv_dumps"
DUMP_STEPS = (0, 1, 16, 64)


def load(layer, genres=("prose", "code", "dialog")):
    qs, ks, vs = [], [], []
    for g in genres:
        with safe_open(str(DUMPS / f"8b_{g}_32768.safetensors"), "pt", device="cuda") as f:
            qs.append(f.get_tensor(f"L{layer}.q"))           # [4, Hq, D]
            ks.append(f.get_tensor(f"L{layer}.k"))           # [Hkv, T, D]
            vs.append(f.get_tensor(f"L{layer}.v"))
    T = min(k.shape[1] for k in ks)
    return torch.stack(qs, 1), torch.stack([k[:, :T] for k in ks]), torch.stack([v[:, :T] for v in vs])


def query_at(qd, t):
    """Linear interpolation between the dumped steps' queries: qd [4, B, Hq, D]."""
    for i in range(len(DUMP_STEPS) - 1):
        a, b = DUMP_STEPS[i], DUMP_STEPS[i + 1]
        if t <= b:
            w = (t - a) / (b - a)
            return ((1 - w) * qd[i].float() + w * qd[i + 1].float()).to(torch.bfloat16)
    return qd[-1]


def read_mask(block_idx, block_cnt, nb):
    B, H, _ = block_idx.shape
    n = block_idx.shape[-1]
    ar = torch.arange(n, device=block_idx.device)
    valid = ar[None, None, :] < block_cnt[..., None]
    m = torch.zeros(B, H, n + 1, dtype=torch.bool, device=block_idx.device)    # column n: unused entries
    m.scatter_(-1, torch.where(valid, block_idx.long(), n), True)
    return m[..., :nb]


@torch.inference_mode()
def run_layer(layer, steps, index=None, calib=None):
    """index: None, "union" or "only" -- KV head 2 on a (random, 5% of blocks) token index --
    or "auto": heads latch onto it by their correction. calib: None or the stage-4 online
    calibration's kind ("logit" / "add"), with no always-exact heads."""
    qd, k, v = load(layer)
    T_end = k.shape[2]
    T0 = T_end - steps
    scale = 128 ** -0.5
    KVSKIP.reset()
    KVSKIP.delta, KVSKIP.max_age, KVSKIP.local, KVSKIP.level = 0.05, 32, 2, 0.99
    KVSKIP.calib, KVSKIP.calib_warm, KVSKIP.calib_hold = (0.3 if calib != "peak" else 0.05, (32, 8), 0) if calib \
        else (None, (), 0)
    KVSKIP.calib_kind = calib or "logit"
    KVSKIP.calib_auto_ix = 0.001 if index == "auto" else None
    KVSKIP.dense_heads = () if calib else ((1, 6), (2, 6))   # one always-exact head on either side
    KVSKIP.index_heads = ((1, 2), (2, 2)) if index in ("union", "only") else ()
    KVSKIP.index_union = index == "union"
    g = torch.Generator(device="cuda").manual_seed(layer)
    KVSKIP.count, KVSKIP.record = False, True
    agree = total = trig_py = trig_fu = 0
    max_err, kept = 0.0, []
    for t in range(steps):
        Tt = T0 + t + 1
        q = query_at(qd, t)
        if index:                                            # union / only / auto
            KVSKIP.idx_blocks = torch.rand(k.shape[0], math.ceil(Tt / BLOCK), device="cuda", generator=g) < 0.05
        o_py = kvskip_attn.refresh_pre_step(1, q, k[:, :, :Tt], v[:, :, :Tt], scale)     # Python planner
        o_fu = kvskip_attn.refresh_fused_step(2, q, k[:, :, :Tt], v[:, :, :Tt], scale)   # device planner
        nb = math.ceil(Tt / BLOCK)
        m_py = KVSKIP.plans[1][-1]
        if t == 0:
            m_fu = torch.ones_like(m_py)
        else:
            f = KVSKIP.state[2]["fused"]
            m_fu = read_mask(f["block_idx"], f["block_cnt"], nb)
            trig_fu += int(f["trig"].sum())
            trig_py += int(m_py.all(-1).sum())
            kept.append(float(m_py.float().mean()))
        same = (m_py == m_fu).all(-1)
        agree += int(same.sum())
        total += same.numel()
        hq = same.repeat_interleave(q.shape[1] // same.shape[1], dim=1)
        if hq.any():
            num = (o_py.float() - o_fu.float()).abs().amax(-1)
            den = o_py.float().abs().amax(-1).clamp_min(1e-6)
            max_err = max(max_err, float((num / den)[hq].max()))
    rate = agree / total
    cal_msg, cal_ok = "", True
    if calib:
        c_py, c_fu = KVSKIP.state[1]["calib"], KVSKIP.state[2]["fused"]["calib"]
        cal_err = float((c_py - c_fu).abs().max())
        cal_ok = cal_err < 0.05 or rate >= 0.99
        cal_msg = (f"; correction: python mean {float(c_py.mean()):.3f} max {float(c_py.max()):.3f}, "
                   f"max |python - device| {cal_err:.1e}")
        if index == "auto":
            a_py, a_fu = KVSKIP.state[1]["aix"], KVSKIP.state[2]["fused"]["aix"].bool()
            cal_msg += f"; latched heads python {int(a_py.sum())} / device {int(a_fu.sum())}"
    print(f"layer {layer:2d}{' index ' + index if index else ''}{' calib ' + calib if calib else ''}: identical read sets "
          f"on {rate:.4f} of head-steps ({agree}/{total}); refreshing "
          f"head-steps python {trig_py} / device {trig_fu}; blocks read {sum(kept) / len(kept):.1%}; "
          f"max rel. output error where identical {max_err:.1e}{cal_msg}")
    KVSKIP.calib, KVSKIP.calib_warm, KVSKIP.calib_kind, KVSKIP.calib_auto_ix = None, (), "logit", None
    return rate >= 0.97 and max_err < 1e-2 and cal_ok


def main():
    steps = 64
    ok = all([run_layer(layer, steps) for layer in (9, 21, 31)]
             + [run_layer(9, steps, index="union"), run_layer(21, steps, index="only")]
             + [run_layer(layer, steps, calib=kind) for kind in ("logit", "add", "peak") for layer in (9, 21, 31)]
             + [run_layer(9, steps, index="union", calib="add"), run_layer(21, steps, index="auto", calib="peak")])
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
