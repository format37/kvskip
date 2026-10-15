"""Stage 3: the layer-major lab -- decode at batch sizes and contexts whose KV does not fit
next to the weights.

Llama-3.1-8B keeps 128 KiB of KV per token: bs=8 at 32k is 32 GiB, more than the 4090
holds next to 15 GB of weights. The lab runs the exact model one layer at a time over a
batch of documents instead ("layer-major"): only one layer's weights and KV are on the
GPU, and the hidden states of all positions carry over from layer to layer.

Per layer:
  1. prefill (exact, shared by all conditions): the prompt positions of every document
     go through the layer (FA2 causal attention, the HF modules for everything else);
     their K/V fill the layer's cache buffer, their outputs are the next layer's input.
  2. decode, per condition: the S teacher-forced decode positions go through the real
     HF decoder layer one step at a time, as in a decode step: the step's K/V is appended
     to the cache and attention runs in the condition's implementation (exact fp32
     reference, stock SDPA, the kvskip kernels + planner). Under teacher forcing a
     layer's decode steps depend only on the same layer's earlier steps (cache, planner
     state) and on the previous layer's outputs, so layer-major order computes exactly
     what step-major decode does. Byte counters are kept on this pass.
  3. timing: the same S layer calls again on the condition's inputs at each batch size
     (B < n: the first B documents; B = n * r: r copies of the cache in distinct memory,
     so the L2 cannot share them), CUDA events around each layer call, no counters.
     The "null" condition (attention returns zeros, reads no KV) times the rest of the
     layer: QKV / o_proj / MLP / norms / cache append. "replay" times the kernels of
     "refresh_pre" alone: its recorded block lists, no planner (what a fused planner can
     reach); its outputs equal refresh_pre's, so it has no KL pass of its own.
     "refresh_cal" (stage 4) is refresh_fused with the online calibration of the estimate
     (--calib*) instead of the always-exact head list.
At the end the decode positions' hidden states give logits -> KL vs the exact reference
(the noise floor is stock SDPA), top-1 agreement with the reference.

A decode step of the whole model at batch B costs the sum over layers of the layer-call
time plus embedding, rotary, final norm and lm_head (timed here too). src/run_batch_e2e.py
checks this composition against the real model where the KV fits (bs=1 at 32k, bs=8 at 4k).

Output: results/lab/<tag>_<ctx>.json (summary) and .npz (per layer and step: times,
counters).

Input: data/docs/{,a4/,s3/}<doc>.json (the stage-3 documents, src/prepare_s3.py), their teacher
tokens data/docs/s3/forced.json (src/make_forced.py), models/llama-3.1-8b, the kvskip extension.
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from flash_attn import flash_attn_func
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.models.llama.modeling_llama import rotate_half

import common
import kvskip_attn
from kvskip_attn import KVSKIP
from prepare_s3 import DOCS

OUT = common.ROOT / "results" / "lab"
FORCED = common.ROOT / "data" / "docs" / "s3" / "forced.json"
DENSE_HEADS = ((2, 5), (5, 2), (16, 5), (27, 1), (26, 3), (25, 1), (31, 2), (6, 7))

# condition -> (attention implementation, kvskip mode)
CONDS = {
    "exact": ("lab_exact", None),
    "sdpa": ("sdpa", None),
    "dense": ("kvskip", "dense"),
    "vskip1": ("kvskip", "vskip1"),
    "refresh": ("kvskip", "refresh"),
    "refresh_pre": ("kvskip", "refresh_pre"),
    "refresh_fused": ("kvskip", "refresh_fused"),
    "refresh_pre_ix": ("kvskip", "refresh_pre"),        # --index-heads on the token index, not always exact
    "refresh_fused_ix": ("kvskip", "refresh_fused"),
    "refresh_cal": ("kvskip", "refresh_fused"),         # stage 4: online calibration, no head list (--calib*)
    "replay": ("kvskip", "replay"),
    "null": ("lab_null", None),
}
UNTIMED = ("exact",)
NO_KL = ("null", "replay")
WARMUP = 8          # decode steps left out of the step-time means (planner state set-up)


class LabCache:
    """One layer's KV: buffers [B, Hkv, cap, D]; update() appends at n and returns views."""

    def __init__(self, k, v, n):
        self.k, self.v, self.n = k, v, n

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        t = key_states.shape[-2]
        self.k[:, :, self.n:self.n + t] = key_states
        self.v[:, :, self.n:self.n + t] = value_states
        self.n += t
        return self.k[:, :, :self.n], self.v[:, :, :self.n]

    def get_seq_length(self, layer_idx=0):
        return self.n


def exact_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """The probe's fp32 decode attention (src/probe_attn.py), batched: [B, Hq, 1, D] -> [B, 1, Hq, D]."""
    B, Hq, _, D = query.shape
    Hkv = key.shape[1]
    out = torch.empty(B, 1, Hq, D, dtype=query.dtype, device=query.device)
    for b in range(B):
        q = query[b, :, 0].float().view(Hkv, Hq // Hkv, D)
        s = torch.einsum("kgd,ktd->kgt", q, key[b].float()) * scaling
        o = torch.einsum("kgt,ktd->kgd", torch.softmax(s, dim=-1), value[b].float())
        out[b, 0] = o.reshape(Hq, D).to(query.dtype)
    return out, None


def null_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    return torch.zeros_like(query).transpose(1, 2), None


def register():
    kvskip_attn.register()
    for name, fn in (("lab_exact", exact_attention), ("lab_null", null_attention)):
        AttentionInterface.register(name, fn)
        AttentionMaskInterface.register(name, sdpa_mask)


def prefill_layer(layer, h, cos, sin, kbuf, vbuf, b, chunk=8192):
    """Exact prefill of one document through one layer, in place: h [T, hidden] bf16."""
    T = h.shape[0]
    attn = layer.self_attn
    D = attn.head_dim
    xn = layer.input_layernorm(h)
    q = attn.q_proj(xn).view(T, -1, D)
    k = attn.k_proj(xn).view(T, -1, D)
    v = attn.v_proj(xn).view(T, -1, D)
    del xn
    c, s = cos[:, None, :], sin[:, None, :]                  # HF apply_rotary_pos_emb, heads on dim 1
    q = (q * c) + (rotate_half(q) * s)
    k = (k * c) + (rotate_half(k) * s)
    kbuf[b, :, :T] = k.transpose(0, 1)
    vbuf[b, :, :T] = v.transpose(0, 1)
    o = flash_attn_func(q[None], k[None], v[None], softmax_scale=attn.scaling, causal=True)[0]
    del q, k, v
    h += attn.o_proj(o.reshape(T, -1))
    del o
    for s0 in range(0, T, chunk):
        x = h[s0:s0 + chunk]
        x += layer.mlp(layer.post_attention_layernorm(x))


def index_steps(ids, P, S, bigram=False):
    """Token-index blocks of every decode step: ids [n, P + S] (prompt + decode inputs); step t's
    current token sits at P + t. Returns S bool tensors [n, nb_t]: blocks holding an earlier
    copy of the current token (or of the token pair, bigram=True) or the token right after one."""
    out = []
    for t in range(S):
        T = P + t + 1
        x = ids[:, :T]
        same = torch.zeros_like(x, dtype=torch.bool)
        same[:, :T - 1] = x[:, :T - 1] == x[:, T - 1:T]
        if bigram:
            prev = torch.zeros_like(same)
            prev[:, 1:T - 1] = x[:, :T - 2] == x[:, T - 2:T - 1]
            same &= prev
        pos = same.clone()
        pos[:, 1:] |= same[:, :T - 1]
        nb = math.ceil(T / 64)
        out.append(torch.nn.functional.pad(pos, (0, nb * 64 - T)).view(-1, nb, 64).any(-1))
    return out


def decode_layer(layer, hid, cache, cos, sin, P, timing=False, idx=None):
    """The S decode steps of one layer, in place: hid [B, S, hidden] inputs -> outputs.
    Step t sits at position P + t and appends its K/V there. idx: per-step token-index blocks
    [B, nb_t] (index heads). Returns per-step ms if timing."""
    S = hid.shape[1]
    ev = []
    for t in range(S):
        cache.n = P + t
        if idx is not None:
            KVSKIP.idx_blocks = idx[t]
        pe = (cos[None, P + t:P + t + 1], sin[None, P + t:P + t + 1])
        if timing:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
        y = layer(hid[:, t:t + 1], attention_mask=None, past_key_values=cache, use_cache=True,
                  position_embeddings=pe)
        if timing:
            b.record()
            ev.append((a, b))
        hid[:, t:t + 1] = y
    if not timing:
        return None
    torch.cuda.synchronize()
    return np.array([a.elapsed_time(b) for a, b in ev])


def set_condition(model, cond, args):
    impl, mode = CONDS[cond]
    model.set_attn_implementation(impl)
    if mode:
        KVSKIP.mode, KVSKIP.tau, KVSKIP.delta, KVSKIP.max_age = mode, args.tau, args.delta, args.max_age
        KVSKIP.dense_heads = DENSE_HEADS if args.dense_heads else ()
        KVSKIP.index_heads = ()
        KVSKIP.index_union = args.index_union
        if cond.endswith("_ix"):
            KVSKIP.index_heads = args.index_heads
            KVSKIP.dense_heads = tuple(h for h in KVSKIP.dense_heads if h not in args.index_heads)
        KVSKIP.dense_layers = (0,)
        KVSKIP.calib, KVSKIP.calib_kind, KVSKIP.calib_warm, KVSKIP.calib_auto_ix = None, "logit", (), None
        if cond == "refresh_cal":
            KVSKIP.dense_heads, KVSKIP.index_heads = (), ()
            KVSKIP.calib, KVSKIP.calib_kind = args.calib, args.calib_kind
            KVSKIP.calib_warm, KVSKIP.calib_auto_ix = tuple(args.calib_warm), args.calib_auto_ix
            KVSKIP.dense_layers = tuple(args.dense_layers)
    KVSKIP.reset()


@torch.inference_mode()
def time_head(model, batches, dev, reps=30):
    """Per-step work outside the decoder layers: embedding, rotary, final norm, lm_head, argmax."""
    emb, rot, norm, head = (model.model.embed_tokens.to(dev), model.model.rotary_emb.to(dev),
                            model.model.norm.to(dev), model.lm_head.to(dev))
    out = {}
    for B in batches:
        ids = torch.randint(0, emb.num_embeddings, (B, 1), device=dev)
        pos = torch.tensor([[1000]], device=dev)
        ms = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            x = emb(ids)
            rot(x, pos)
            head(norm(x)).argmax(-1)
            b.record()
            ms.append((a, b))
        torch.cuda.synchronize()
        out[B] = float(np.median([a.elapsed_time(b) for a, b in ms][5:]))
    return out


@torch.inference_mode()
def run(args):
    register()
    dev = "cuda"
    path = common.MODELS["8b"]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu",
                                                 attn_implementation="sdpa").eval()
    cfg = model.config
    L, Hkv, D = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    forced = json.loads(FORCED.read_text())
    docs, S, n = args.docs, args.steps, len(args.docs)
    P0 = args.ctx - S                                       # prompt tokens, as in stages 1-2
    prompts = [common.exact_prompt(tok, common.load_named_doc(d), P0) for d in docs]
    P = P0 - 1                                              # prefill; decode step t sits at P + t
    pre_ids = torch.tensor([p[:-1] for p in prompts], device=dev)
    dec_ids = torch.tensor([[p[-1]] + forced[d]["tokens"][:S - 1] for d, p in zip(docs, prompts)], device=dev)
    conds = ["exact"] + [c for c in args.conds if c != "exact"]
    batches = sorted(args.batches)
    reps = max(1, math.ceil(max(batches) / n))
    t_start = time.perf_counter()
    print(f"[lab] ctx={args.ctx} docs={n} steps={S} prefill={P} batches={batches} conds={conds}", flush=True)

    emb = model.model.embed_tokens.to(dev)
    h_pre = emb(pre_ids)                                    # [n, P, hidden]
    h0 = emb(dec_ids)                                       # [n, S, hidden]
    emb.to("cpu")
    hid = {c: h0.clone() for c in conds if c not in NO_KL}
    idx = None
    if any(c.endswith("_ix") for c in conds) or ("refresh_cal" in conds and args.calib_auto_ix is not None):
        idx = index_steps(torch.cat([pre_ids, dec_ids], 1), P, S, bigram=args.index_kind == "bigram")
    hid_null = h0.clone()
    rot = model.model.rotary_emb.to(dev)
    cos, sin = rot(h0, torch.arange(P + S, device=dev)[None])
    cos, sin = cos[0], sin[0]
    kbuf = torch.empty(n * reps, Hkv, P + S, D, dtype=torch.bfloat16, device=dev)
    vbuf = torch.empty_like(kbuf)

    layers = range(L) if not args.layers else args.layers
    times = {c: {B: np.full((L, S), np.nan) for B in batches} for c in conds if c not in UNTIMED}
    counters = {c: {} for c in conds}
    t_prefill = 0.0
    for l in layers:
        layer = model.model.layers[l].to(dev)
        t0 = time.perf_counter()
        for b in range(n):
            prefill_layer(layer, h_pre[b], cos[:P], sin[:P], kbuf, vbuf, b)
        for r in range(1, reps):
            kbuf[r * n:(r + 1) * n, :, :P] = kbuf[:n, :, :P]
            vbuf[r * n:(r + 1) * n, :, :P] = vbuf[:n, :, :P]
        torch.cuda.synchronize()
        t_prefill += time.perf_counter() - t0
        t0 = time.perf_counter()
        rec_masks = keep_pre = None
        for c in conds:
            set_condition(model, c, args)
            if c == "replay":
                if rec_masks is None:
                    continue
                x_keep = keep_pre                           # refresh_pre's inputs and block lists
            else:
                x_in = hid[c] if c not in NO_KL else hid_null
                if c not in UNTIMED:
                    x_keep = x_in.clone()
            ix = idx if c.endswith("_ix") or (c == "refresh_cal" and args.calib_auto_ix is not None) else None
            if c not in NO_KL:                              # main pass: outputs + byte counters
                KVSKIP.count, KVSKIP.record = True, c == "refresh_pre"
                decode_layer(layer, x_in, LabCache(kbuf[:n], vbuf[:n], P), cos, sin, P, idx=ix)
                KVSKIP.record = False
                tot = kvskip_attn.log_totals()
                for key, val in tot.items():
                    counters[c].setdefault(key, np.zeros(L))[l] = val
                if c == "refresh_pre" and l in KVSKIP.plans:
                    rec_masks, keep_pre = KVSKIP.plans.pop(l), x_keep
            if c in UNTIMED:
                continue
            KVSKIP.count = False
            for B in batches:
                xb = x_keep.repeat(B // n, 1, 1) if B >= n else x_keep[:B].clone()
                KVSKIP.reset()
                if c == "replay":
                    KVSKIP.plans[l] = [kvskip_attn.K.make_plan(m.repeat(B // n, 1, 1) if B >= n else m[:B],
                                                               sync=False) for m in rec_masks]
                ixb = None if ix is None else [m.repeat(B // n, 1) if B >= n else m[:B] for m in ix]
                times[c][B][l] = decode_layer(layer, xb, LabCache(kbuf[:B], vbuf[:B], P), cos, sin, P,
                                              timing=True, idx=ixb)
                del xb
            if c == "null":
                hid_null = x_keep                           # null attention: inputs unchanged
            if c == "replay":
                KVSKIP.plans.pop(l, None)
                rec_masks = keep_pre = None
        layer.to("cpu")
        torch.cuda.empty_cache()
        msg = " ".join(f"{c}={np.nanmean(times[c][max(batches)][l, WARMUP:]):.2f}" for c in times)
        print(f"  layer {l:2d}: prefill+conds {time.perf_counter() - t0:5.1f}s | ms/layer-step at "
              f"B={max(batches)}: {msg}", flush=True)
    del kbuf, vbuf, h_pre
    torch.cuda.empty_cache()

    # logits at the decode positions -> KL vs the exact reference, top-1 agreement
    norm, head = model.model.norm.to(dev), model.lm_head.to(dev)
    ref = torch.log_softmax(head(norm(hid["exact"])).float(), -1)
    ref_top = ref.argmax(-1)
    kl, top1 = {}, {}
    for c in hid:
        if c == "exact":
            continue
        lp = torch.log_softmax(head(norm(hid[c])).float(), -1)
        kl[c] = (ref.exp() * (ref - lp)).sum(-1).cpu().numpy()            # [n, S]
        top1[c] = (lp.argmax(-1) == ref_top).float().cpu().numpy()
        del lp
    forced_match = (ref_top[:, :-1] == dec_ids[:, 1:]).float().mean().item()
    head_ms = time_head(model, batches, dev)

    res = {"ctx": args.ctx, "steps": S, "docs": docs, "prefill_tokens": P, "batches": batches,
           "dense_heads": [list(h) for h in DENSE_HEADS] if args.dense_heads else [],
           "index_heads": [list(h) for h in args.index_heads], "index_kind": args.index_kind,
           "index_union": args.index_union,
           "calib": {"rate": args.calib, "kind": args.calib_kind, "warm": args.calib_warm,
                     "auto_ix": args.calib_auto_ix, "dense_layers": args.dense_layers},
           "delta": args.delta, "max_age": args.max_age, "tau": args.tau, "warmup": WARMUP,
           "layers": list(layers), "ref_argmax_eq_forced": forced_match, "head_ms": head_ms,
           "prefill_s": round(t_prefill, 1), "total_s": round(time.perf_counter() - t_start, 1),
           "kl": {}, "step_ms": {}, "attn_ms": {}, "bytes": {}}
    if "sdpa" in kl:
        noise = kl["sdpa"].mean(1)                                          # per document
        for c in kl:
            res["kl"][c] = {"kl_mean": float(kl[c].mean()), "x_noise_by_doc": (kl[c].mean(1) / noise).tolist(),
                            "x_noise": float(kl[c].mean() / kl["sdpa"].mean()),
                            "p95": float(np.quantile(kl[c], 0.95)), "top1": float(top1[c].mean())}
    for c in times:
        for B in batches:
            per_layer = np.nanmean(times[c][B][:, WARMUP:], axis=1)          # [L]
            res["step_ms"].setdefault(c, {})[B] = float(np.nansum(per_layer) + head_ms[B])
            if "null" in times:
                null = np.nanmean(times["null"][B][:, WARMUP:], axis=1)
                res["attn_ms"].setdefault(c, {})[B] = float(np.nansum(per_layer - null))
    for c, ctr in counters.items():
        if "blocks_total" in ctr:
            tot = ctr["blocks_total"].sum()
            if "blocks_read" in ctr:                                        # K and V of the same blocks
                res["bytes"][c] = {"kv_read_frac_sparse_layers": float(ctr["blocks_read"].sum() / tot),
                                   "refresh_rate": float(ctr.get("dense_heads", np.zeros(1)).sum()
                                                         / max(1.0, ctr.get("masked_heads", np.ones(1)).sum()))}
            elif "v_tiles" in ctr:                                          # K in full, V partly
                res["bytes"][c] = {"v_read_frac_sparse_layers": float(ctr["v_tiles"].sum() / tot)}
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{args.tag}_{args.ctx}"
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1))
    arrays = {f"t_{c}_B{B}": times[c][B] for c in times for B in batches}
    arrays.update({f"kl_{c}": v for c, v in kl.items()})
    arrays.update({f"ctr_{c}_{k}": v for c, ctr in counters.items() for k, v in ctr.items()})
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    print(f"[lab] done in {res['total_s']}s (prefill {res['prefill_s']}s); reference argmax == forced "
          f"{forced_match:.3f}", flush=True)
    for c in times:
        print(f"  {c:12s} " + " ".join(f"B={B}: {res['step_ms'][c][B]:7.1f} ms" for B in batches)
              + (f" | KL x{res['kl'][c]['x_noise']:.2f}" if c in res["kl"] else "")
              + (f" | {res['bytes'][c]}" if c in res["bytes"] else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--docs", nargs="+", default=list(DOCS))
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--conds", nargs="+",
                    default=["sdpa", "dense", "vskip1", "refresh", "refresh_pre", "replay", "null"])
    ap.add_argument("--layers", nargs="*", type=int, default=[], help="subset of layers (quick tests)")
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--max-age", type=int, default=32)
    ap.add_argument("--no-dense-heads", dest="dense_heads", action="store_false")
    ap.add_argument("--index-heads", nargs="*", default=["2.5"],
                    help="layer.kv_head pairs on the token index in the *_ix conditions")
    ap.add_argument("--index-kind", default="token", choices=["token", "bigram"])
    ap.add_argument("--index-union", action="store_true",
                    help="index blocks join the stored set (trigger stays) instead of replacing it")
    ap.add_argument("--calib", type=float, default=0.05, help="refresh_cal: rate of the online calibration")
    ap.add_argument("--calib-kind", default="peak", choices=["logit", "add", "peak"])
    ap.add_argument("--calib-warm", nargs="*", type=int, default=[32, 8], help="W A: max age A in the first W steps")
    ap.add_argument("--calib-auto-ix", type=float, default=None, help="heads above this correction join the index")
    ap.add_argument("--dense-layers", nargs="*", type=int, default=[0], help="refresh_cal: dense layers")
    ap.add_argument("--tag", default="s3")
    a = ap.parse_args()
    a.index_heads = tuple(tuple(int(x) for x in lh.split(".")) for lh in a.index_heads)
    run(a)


if __name__ == "__main__":
    main()
