"""HF attention implementation "kvskip": decode attention on the kvskip CUDA kernels.

Prefill (q_len > 1) goes to SDPA. Decode (q_len == 1) is selected by KVSKIP.mode:
  "dense"    dense split-K kernel (bf16 in/out, fp32 accumulation)
  "vskip2"   two-pass V-skip: pass 1 reads all of K and writes P; GQA-union top-p
             `level` per step; pass 2 reads P and V of the kept blocks
  "vskip1"   single-pass V-skip at threshold `tau`; the normaliser estimate is the
             previous step's log-sum-exp of the same layer (first step: sink bound)
  "vskip1sink" the same with the sink-anchored lower bound on every step
  "refresh"  K+V skip: attend the stored GQA-union top-p set + sink + `local` newest
             blocks; per KV head, the centroid-drift estimate of the missed mass
             (src/reuse.py "estd") decides a refresh = dense pass with block stats,
             which also yields the new set. Same semantics as the probe's
             ReuseConfig("estd", delta, max_age) with exact refresh.
Layers in KVSKIP.dense_layers always use the dense kernel.

GrowingCache() is a preallocated KV cache whose layers return views into a
buffer grown by doubling, so decode does not pay HF DynamicCache's per-step
torch.cat of the whole cache (needed for honest end-to-end timing).

The planner is plain PyTorch on the GPU (several small launches per layer and
step); a production version would fuse it into the kernels.
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AttentionInterface
from transformers.cache_utils import Cache, DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kernels import kvskip as K  # noqa: E402

BLOCK = 64


class GrowingLayer(DynamicLayer):
    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        self._k = self._v = None
        self._n = 0

    def _ensure(self, need: int, ref: torch.Tensor):
        cap = 0 if self._k is None else self._k.shape[-2]
        if need <= cap:
            return
        new_cap = math.ceil(max(need + 512, 2 * cap) / BLOCK) * BLOCK
        B, H, _, D = ref.shape
        k = torch.empty(B, H, new_cap, D, dtype=ref.dtype, device=ref.device)
        v = torch.empty_like(k)
        if self._n:
            k[:, :, :self._n] = self._k[:, :, :self._n]
            v[:, :, :self._n] = self._v[:, :, :self._n]
        self._k, self._v = k, v

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        t = key_states.shape[-2]
        self._ensure(self._n + t, key_states)
        self._k[:, :, self._n:self._n + t] = key_states
        self._v[:, :, self._n:self._n + t] = value_states
        self._n += t
        self.keys, self.values = self._k[:, :, :self._n], self._v[:, :, :self._n]
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self._n if self.is_initialized else 0

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove > 0:          # legacy: absolute target length
            tokens_to_remove = max(0, self._n - tokens_to_remove)
        self._n -= abs(tokens_to_remove)
        self.keys, self.values = self._k[:, :, :self._n], self._v[:, :, :self._n]


def GrowingCache() -> Cache:
    return Cache(layer_class_to_replicate=GrowingLayer)


@dataclass
class KvskipConfig:
    mode: str = "dense"
    level: float = 0.99           # top-p mass of the selected / stored sets
    tau: float = 1e-4             # vskip1 threshold
    delta: float = 0.05           # refresh trigger on the estimated missed mass
    max_age: int = 32
    local: int = 2
    dense_layers: tuple = (0,)
    dense_heads: tuple = ()       # (layer, kv_head) pairs refreshed (= exact) on every step
    state: dict = field(default_factory=dict)       # per-layer planner state (refresh mode)
    log: dict = field(default_factory=dict)         # per-layer counters

    def reset(self):
        self.state.clear()
        self.log.clear()


KVSKIP = KvskipConfig()


def _count(layer, key, value):
    d = KVSKIP.log.setdefault(layer, {})
    d[key] = d.get(key, 0.0) + value


def _topp_union(logw: torch.Tensor, G: int, level: float, local: int) -> torch.Tensor:
    return K.select_blocks(logw, G, "topp", level, local)


def _centroid_sums(k: torch.Tensor) -> torch.Tensor:
    """k [B, Hkv, T, D] -> per-block sums of keys [B, Hkv, nb, D] fp32."""
    B, H, T, D = k.shape
    nb = math.ceil(T / BLOCK)
    kp = torch.nn.functional.pad(k.float(), (0, 0, 0, nb * BLOCK - T))
    return kp.view(B, H, nb, BLOCK, D).sum(3)


def _refresh_state(st, q32, logw, blk_max, csum, T, G, heads=None):
    """(Re)initialise the stored set and the estimate references for all or some KV heads."""
    B, Hq, nb = logw.shape
    keep = _topp_union(logw, G, KVSKIP.level, KVSKIP.local)            # [B, Hkv, nb]
    cs = _centroid_scores(q32, csum, T, st["scale"])                   # [B, Hq, nb]
    if heads is None:
        st.update(keep=keep, lse_ref=logw, cs_ref=cs,
                  age=torch.zeros(keep.shape[:2], dtype=torch.long, device=keep.device))
        return
    hq = heads.repeat_interleave(G, dim=1)                             # [B, Hq]
    st["keep"] = _pad_blocks(st["keep"], nb, False)
    st["lse_ref"] = _pad_blocks(st["lse_ref"], nb, float("-inf"))
    st["cs_ref"] = _pad_blocks(st["cs_ref"], nb, 0.0)
    st["keep"] = torch.where(heads[..., None], keep, st["keep"])
    st["lse_ref"] = torch.where(hq[..., None], logw, st["lse_ref"])
    st["cs_ref"] = torch.where(hq[..., None], cs, st["cs_ref"])
    st["age"] = torch.where(heads, torch.zeros_like(st["age"]), st["age"])


def _centroid_scores(q32, csum, T, scale):
    nb = csum.shape[2]
    sizes = torch.full((nb,), float(BLOCK), device=csum.device)
    sizes[-1] = T - BLOCK * (nb - 1)
    c = csum / sizes[None, None, :, None]                              # [B, Hkv, nb, D]
    B, Hq, D = q32.shape
    Hkv = c.shape[1]
    return torch.einsum("bkgd,bknd->bkgn", q32.view(B, Hkv, Hq // Hkv, D), c).reshape(B, Hq, nb) * scale


def _pad_blocks(x, nb, value):
    return torch.nn.functional.pad(x, (0, nb - x.shape[-1]), value=value) if x.shape[-1] < nb else x


def refresh_step(layer, q, k, v, scale):
    """One decode step of the refresh policy for one layer. q [B, Hq, D]; k, v [B, Hkv, T, D]."""
    B, Hq, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    G = Hq // Hkv
    nb = math.ceil(T / BLOCK)
    q32 = q.float()
    st = KVSKIP.state.get(layer)
    if st is None:                                       # first decode step: dense + stats
        r = K.decode(q, k, v, scale, stats=True)
        st = {"scale": scale, "csum": _centroid_sums(k), "T": T}
        KVSKIP.state[layer] = st
        _refresh_state(st, q32, r["blk_max"] + torch.log(r["blk_sum"]), r["blk_max"], st["csum"], T, G)
        _count(layer, "dense_heads", B * Hkv)
        _count(layer, "blocks_read", B * Hkv * nb)
        _count(layer, "blocks_total", B * Hkv * nb)
        return r["out"]

    # keep the per-block key sums current: add the tokens appended since the last step
    csum = _pad_blocks(st["csum"].transpose(2, 3), nb, 0.0).transpose(2, 3)   # [B, Hkv, nb, D]
    for t in range(st["T"], T):
        csum[:, :, t // BLOCK] += k[:, :, t].float()
    st["csum"], st["T"] = csum, T

    always = K.always_keep(nb, KVSKIP.local, q.device)
    stored = _pad_blocks(st["keep"], nb, False)
    sel = stored | always
    plan = K.make_plan(sel, sync=False)
    r = K.decode(q, k, v, scale, plan=plan)
    out, lse_att = r["out"], r["lse"]                                  # lse over the attended set

    # centroid-drift estimate of the mass outside `sel`, max over the G heads of a KV head
    nbr = st["lse_ref"].shape[-1]
    cs = _centroid_scores(q32, csum, T, scale)                         # [B, Hq, nb]
    est = st["lse_ref"] + cs[..., :nbr] - st["cs_ref"][..., :nbr]
    est = _pad_blocks(est, nb, float("-inf"))
    unsel = ~sel
    unsel[..., nbr:] = False
    est = est.masked_fill(~unsel.repeat_interleave(G, dim=1), float("-inf"))
    miss = torch.sigmoid(est.logsumexp(-1) - lse_att).view(B, Hkv, G).amax(-1)   # [B, Hkv]
    st["age"] += 1
    trig = (miss > KVSKIP.delta) | (st["age"] >= KVSKIP.max_age)
    forced = [h for (l, h) in KVSKIP.dense_heads if l == layer]
    if forced:
        trig[:, forced] = True
    n_trig = int(trig.sum())
    _count(layer, "blocks_read", float(sel.sum()))
    _count(layer, "blocks_total", B * Hkv * nb)
    _count(layer, "masked_heads", B * Hkv)
    if n_trig:
        # dense + stats only for the triggered KV heads: a plan with all blocks for them,
        # none for the others (their CTAs exit at once)
        full = trig[..., None].expand(B, Hkv, nb).contiguous()
        r2 = K.decode(q, k, v, scale, plan=K.make_plan(full, sync=False), stats=True)
        hq = trig.repeat_interleave(G, dim=1)                          # [B, Hq]
        out = torch.where(hq[..., None], r2["out"], out)
        logw = r2["blk_max"] + torch.log(r2["blk_sum"])
        _refresh_state(st, q32, _pad_blocks(logw, nb, float("-inf")), r2["blk_max"], csum, T, G, heads=trig)
        _count(layer, "dense_heads", float(n_trig))
        _count(layer, "blocks_read", float(n_trig * nb))
    return out


def kvskip_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if query.shape[2] != 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    layer = module.layer_idx
    q = query[:, :, 0]                                                 # [B, Hq, D]
    mode = KVSKIP.mode if layer not in KVSKIP.dense_layers else "dense"
    if mode == "dense":
        out = K.decode(q, key, value, scaling)["out"]
    elif mode == "vskip2":
        out = K.vskip_two_pass(q, key, value, scaling, rule="topp", value=KVSKIP.level,
                               local=KVSKIP.local, sync=False)["out"]
    elif mode in ("vskip1", "vskip1sink"):
        st = KVSKIP.state.setdefault(layer, {})
        prev = st.get("lse") if mode == "vskip1" else None
        r = K.vskip_one_pass(q, key, value, scaling, tau=KVSKIP.tau, local=KVSKIP.local, lse_prev=prev)
        st["lse"] = r["lse"]
        out = r["out"]
    elif mode == "refresh":
        out = refresh_step(layer, q, key, value, scaling)
    else:
        raise ValueError(mode)
    return out[:, None], None                                          # [B, 1, Hq, D]


def register():
    AttentionInterface.register("kvskip", kvskip_attention_forward)
    AttentionMaskInterface.register("kvskip", sdpa_mask)
