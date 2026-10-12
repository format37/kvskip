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
             ReuseConfig("estd", delta, max_age) with exact refresh. The trigger is
             evaluated after the masked pass, so a triggered head is read twice.
  "refresh_pre" the same policy with the trigger evaluated before the pass (stage 3):
             the estimate needs only q, the block key centroids, the refresh-step
             block log-sum-exps and the attended set's log-sum-exp of the previous
             step, so one masked kernel with block stats serves both kinds of heads
             (triggered: every block, others: the stored set) and nothing waits on
             the host.
  "refresh_fused" refresh_pre with the planner in two CUDA kernels (kernels/kvskip_planner.cu):
             plan (estimate, decision, block list) -> masked + stats attention ->
             update (top-p union, references, attended lse); three launches per layer
             and step, no host sync. The first step initialises through refresh_pre.
  "replay"   the masked + stats kernel on block lists recorded from a "refresh_pre"
             run (KVSKIP.plans[layer], one Plan per step): the kernels of the policy
             without its planner, i.e. what fusing the planner into kernels can reach.
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
    capacity = 0                  # first allocation in tokens (GrowingCache(capacity=...))

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        self._k = self._v = None
        self._n = 0

    def _ensure(self, need: int, ref: torch.Tensor):
        cap = 0 if self._k is None else self._k.shape[-2]
        if need <= cap:
            return
        new_cap = math.ceil(max(need + 512, 2 * cap, self.capacity) / BLOCK) * BLOCK
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


def GrowingCache(capacity: int = 0) -> Cache:
    """capacity: tokens to allocate up front (no doubling below it)."""
    layer = GrowingLayer if not capacity else type("GrowingLayerCap", (GrowingLayer,), {"capacity": capacity})
    return Cache(layer_class_to_replicate=layer)


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
    index_heads: tuple = ()       # (layer, kv_head) pairs on the token index (refresh_pre / refresh_fused)
    index_union: bool = False     # index blocks join the head's attended set (trigger stays) instead of replacing it
    idx_blocks: object = None     # this step's token-index blocks [B, >= nb] bool, set by the driver
    count: bool = True            # keep the per-layer byte counters (GPU tensors, no host sync)
    record: bool = False          # refresh_pre: append each step's block mask to plans[layer]
    state: dict = field(default_factory=dict)       # per-layer planner state (refresh mode)
    log: dict = field(default_factory=dict)         # per-layer counters
    plans: dict = field(default_factory=dict)       # per-layer recorded masks / Plans ("replay")

    def reset(self):
        self.state.clear()
        self.log.clear()


KVSKIP = KvskipConfig()


def _count(layer, key, value):
    if not KVSKIP.count:
        return
    d = KVSKIP.log.setdefault(layer, {})
    d[key] = d.get(key, 0.0) + value


def log_totals() -> dict:
    """Counters summed over layers, as floats (one host sync)."""
    out = {}
    for d in KVSKIP.log.values():
        for key, val in d.items():
            out[key] = out.get(key, 0.0) + float(val)
    return out


def _topp_union(logw: torch.Tensor, G: int, level: float, local: int) -> torch.Tensor:
    return K.select_blocks(logw, G, "topp", level, local)


def _centroid_sums(k: torch.Tensor) -> torch.Tensor:
    """k [B, Hkv, T, D] -> per-block sums of keys [B, Hkv, nb, D] fp32 (no fp32 copy of k)."""
    B, H, T, D = k.shape
    nb, full = math.ceil(T / BLOCK), T // BLOCK
    out = torch.zeros(B, H, nb, D, dtype=torch.float32, device=k.device)
    if full:
        out[:, :, :full] = k[:, :, :full * BLOCK].unflatten(2, (full, BLOCK)).sum(3, dtype=torch.float32)
    if T > full * BLOCK:
        out[:, :, full] = k[:, :, full * BLOCK:].sum(2, dtype=torch.float32)
    return out


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

    csum = _update_csum(st, k, T, nb)
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
    _count(layer, "blocks_read", sel.sum())
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


def _update_csum(st, k, T, nb):
    """Keep the per-block key sums current: add the tokens appended since the last step."""
    csum = _pad_blocks(st["csum"].transpose(2, 3), nb, 0.0).transpose(2, 3)   # [B, Hkv, nb, D]
    for t in range(st["T"], T):
        csum[:, :, t // BLOCK] += k[:, :, t].float()
    st["csum"], st["T"] = csum, T
    return csum


def _attended_lse(logw, keep, G):
    """log-sum-exp of the block log-weights [B, Hq, nb] over the blocks in keep [B, Hkv, nb]."""
    return logw.masked_fill(~keep.repeat_interleave(G, dim=1), float("-inf")).logsumexp(-1)


def refresh_pre_step(layer, q, k, v, scale):
    """refresh_step with the trigger decided before the pass: one masked kernel with block
    stats per step, no host sync. The attended mass in the estimate's denominator is the
    previous step's log-sum-exp over the (current) stored set, from that step's stats."""
    B, Hq, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    G = Hq // Hkv
    nb = math.ceil(T / BLOCK)
    q32 = q.float()
    always = K.always_keep(nb, KVSKIP.local, q.device)
    st = KVSKIP.state.get(layer)
    if st is None:                                       # first decode step: dense + stats
        r = K.decode(q, k, v, scale, stats=True)
        st = {"scale": scale, "csum": _centroid_sums(k), "T": T}
        KVSKIP.state[layer] = st
        logw = r["blk_max"] + torch.log(r["blk_sum"])
        _refresh_state(st, q32, logw, r["blk_max"], st["csum"], T, G)
        st["lse_att"] = _attended_lse(logw, st["keep"] | always, G)
        if KVSKIP.record:
            KVSKIP.plans.setdefault(layer, []).append(torch.ones(B, Hkv, nb, dtype=torch.bool, device=q.device))
        _count(layer, "dense_heads", B * Hkv)
        _count(layer, "blocks_read", B * Hkv * nb)
        _count(layer, "blocks_total", B * Hkv * nb)
        return r["out"]

    csum = _update_csum(st, k, T, nb)
    stored = _pad_blocks(st["keep"], nb, False)
    sel = stored | always
    ix = [h for (l, h) in KVSKIP.index_heads if l == layer]
    if ix and KVSKIP.index_union:
        sel[:, ix] |= KVSKIP.idx_blocks[:, None, :nb]
    nbr = st["lse_ref"].shape[-1]
    cs = _centroid_scores(q32, csum, T, scale)                         # [B, Hq, nb]
    est = _pad_blocks(st["lse_ref"] + cs[..., :nbr] - st["cs_ref"][..., :nbr], nb, float("-inf"))
    unsel = ~sel
    unsel[..., nbr:] = False
    est = est.masked_fill(~unsel.repeat_interleave(G, dim=1), float("-inf"))
    miss = torch.sigmoid(est.logsumexp(-1) - st["lse_att"]).view(B, Hkv, G).amax(-1)   # [B, Hkv]
    st["age"] += 1
    trig = (miss > KVSKIP.delta) | (st["age"] >= KVSKIP.max_age)
    forced = [h for (l, h) in KVSKIP.dense_heads if l == layer]
    if forced:
        trig[:, forced] = True
    read = sel | trig[..., None]                                       # triggered heads: every block
    if ix and not KVSKIP.index_union:                                  # token-index heads: never refresh
        trig[:, ix] = False
        read[:, ix] = (KVSKIP.idx_blocks[:, :nb] | always)[:, None, :]
    if KVSKIP.record:
        KVSKIP.plans.setdefault(layer, []).append(read)
    r = K.decode(q, k, v, scale, plan=K.make_plan(read, sync=False), stats=True)
    logw = r["blk_max"] + torch.log(r["blk_sum"])                      # -inf on blocks not read
    _refresh_state(st, q32, logw, r["blk_max"], csum, T, G, heads=trig)
    st["lse_att"] = _attended_lse(logw, _pad_blocks(st["keep"], nb, False) | always, G)
    _count(layer, "blocks_read", read.sum())
    _count(layer, "blocks_total", B * Hkv * nb)
    _count(layer, "masked_heads", B * Hkv)
    _count(layer, "dense_heads", trig.sum())
    return r["out"]


FUSED_MARGIN = 64                 # spare block capacity of the fused planner state (4096 tokens)


def _pack_fused(st, layer, q, nb):
    """Device-planner state from refresh_pre's first-step Python state, with spare capacity."""
    B, Hq, D = q.shape
    Hkv = st["keep"].shape[1]
    NBC = nb + FUSED_MARGIN
    dev = q.device

    def padded(x, fill, dtype):
        out = torch.full((*x.shape[:2], NBC, *x.shape[3:]), fill, dtype=dtype, device=dev)
        out[:, :, :x.shape[2]] = x
        return out
    flags = torch.zeros(Hkv, dtype=torch.int32, device=dev)
    for (l, h) in KVSKIP.dense_heads:
        if l == layer:
            flags[h] = 1
    for (l, h) in KVSKIP.index_heads:
        if l == layer:
            flags[h] = 4 if KVSKIP.index_union else 2
    st["fused"] = dict(
        NBC=NBC, flags=flags,
        keep=padded(_pad_blocks(st["keep"], nb, False), 0, torch.uint8),
        lse_ref=padded(_pad_blocks(st["lse_ref"], nb, float("-inf")), float("-inf"), torch.float32),
        cs_ref=padded(_pad_blocks(st["cs_ref"], nb, 0.0), 0.0, torch.float32),
        csum=padded(st["csum"], 0.0, torch.float32),
        lse_att=st["lse_att"].float().contiguous(), age=st["age"].to(torch.int32).contiguous(),
        cs_now=torch.zeros(B, Hq, NBC, dtype=torch.float32, device=dev),
        trig=torch.zeros(B, Hkv, dtype=torch.uint8, device=dev),
        block_idx=torch.zeros(B, Hkv, NBC, dtype=torch.int32, device=dev),
        block_cnt=torch.zeros(B, Hkv, dtype=torch.int32, device=dev))


def _grow_fused(f, NBC):
    fills = dict(keep=0, lse_ref=float("-inf"), cs_ref=0.0, csum=0.0, cs_now=0.0, block_idx=0)
    for key, fill in fills.items():
        x = f[key]
        out = torch.full((*x.shape[:2], NBC, *x.shape[3:]), fill, dtype=x.dtype, device=x.device)
        out[:, :, :x.shape[2]] = x
        f[key] = out
    f["NBC"] = NBC


def refresh_fused_step(layer, q, k, v, scale):
    st = KVSKIP.state.get(layer)
    if st is None:
        out = refresh_pre_step(layer, q, k, v, scale)
        _pack_fused(KVSKIP.state[layer], layer, q, math.ceil(k.shape[2] / BLOCK))
        return out
    f = st["fused"]
    B, Hkv, T = k.shape[0], k.shape[1], k.shape[2]
    nb = math.ceil(T / BLOCK)
    if nb > f["NBC"]:
        _grow_fused(f, nb + FUSED_MARGIN)
    ext = K.ext()
    idx = None
    if any(l == layer for (l, _) in KVSKIP.index_heads):
        idx = torch.zeros(B, f["NBC"], dtype=torch.uint8, device=q.device)
        idx[:, :nb] = KVSKIP.idx_blocks[:, :nb]
    ext.plan(q, k, f["csum"], f["lse_ref"], f["cs_ref"], f["keep"], f["lse_att"], f["age"], f["flags"],
             f["cs_now"], f["trig"], f["block_idx"], f["block_cnt"], float(scale), float(KVSKIP.delta),
             int(KVSKIP.max_age), int(KVSKIP.local), True, idx)
    r = K.decode(q, k, v, scale, plan=K.Plan(f["block_idx"], f["block_cnt"], nb, nb), stats=True)
    ext.update(r["blk_max"], r["blk_sum"], f["trig"], f["cs_now"], f["keep"], f["lse_ref"], f["cs_ref"],
               f["lse_att"], T, float(KVSKIP.level), int(KVSKIP.local))
    _count(layer, "blocks_read", f["block_cnt"].sum())
    _count(layer, "blocks_total", B * Hkv * nb)
    _count(layer, "masked_heads", B * Hkv)
    _count(layer, "dense_heads", f["trig"].sum())
    return r["out"]


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
        r = K.vskip_one_pass(q, key, value, scaling, tau=KVSKIP.tau, local=KVSKIP.local, lse_prev=prev,
                             count=KVSKIP.count, sync=False)
        st["lse"] = r["lse"]
        out = r["out"]
        if KVSKIP.count:
            _count(layer, "v_tiles", r["v_tiles"].sum())
            _count(layer, "blocks_total", r["v_tiles"].numel() * math.ceil(key.shape[2] / BLOCK))
    elif mode == "refresh":
        out = refresh_step(layer, q, key, value, scaling)
    elif mode == "refresh_pre":
        out = refresh_pre_step(layer, q, key, value, scaling)
    elif mode == "refresh_fused":
        out = refresh_fused_step(layer, q, key, value, scaling)
    elif mode == "replay":
        st = KVSKIP.state.setdefault(layer, {"t": 0})
        out = K.decode(q, key, value, scaling, plan=KVSKIP.plans[layer][st["t"]], stats=True)["out"]
        st["t"] += 1
    else:
        raise ValueError(mode)
    return out[:, None], None                                          # [B, 1, Hq, D]


def register():
    AttentionInterface.register("kvskip", kvskip_attention_forward)
    AttentionMaskInterface.register("kvskip", sdpa_mask)
