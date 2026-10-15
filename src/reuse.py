"""Stage 2, track A: block-selection reuse (A1) and mask locality (A2) inside the probe.

A1 -- what a "K+V-skip with periodic refresh" kernel would compute. Per layer and
KV head the probe keeps a stored block set:
  * refresh step: the exact weights of this step give the GQA-union top-p set
    (`level` of each query head's mass); with refresh="exact" this step itself
    is computed densely (a kernel does one dense pass that also emits per-block
    stats), with refresh="masked" the fresh set is applied immediately;
  * other steps: only stored set + block 0 (sink) + the `local` newest blocks
    are attended (renormalised over them), i.e. neither K nor V of the other
    blocks is read.
Refresh policies: "fresh" (new set every step, K fully read = the two-pass
V-skip kernel), "timer" (every N steps, all heads at once), "oracle" (a head
refreshes when its stale set covers < theta of the exact mass -- needs the full
attention, so it is an upper bound for any trigger) and "drift" (a head
refreshes when its query moved: max over the G heads of 1 - cos(q, q_at_refresh)
> delta; costs nothing, the query is in registers anyway).
"estc" / "estd" refresh a head when a cheap estimate of the mass its stale set
misses exceeds delta. Both read only per-block key centroids (1/64 of K):
  estc: block sumexp ~ n_b * exp(q . c_b * scale)          (Jensen lower bound)
  estd: block sumexp ~ S_b(refresh) * exp((q - q_r) . c_b * scale)
        (exact block log-sum-exp of the refresh step, moved by the query drift
        projected on the centroid); the attended part is exact (the masked pass
        knows its own m and l).
"estp" (stage 3) is estd with the trigger decided before the attention pass: the
attended set's log-sum-exp in the estimate is the previous step's (over the current
stored set), so a kernel can choose dense or masked per head before reading K.
Layers in `dense_layers` are always dense (layer 0 per the stage-1 finding).
Heads in `index_heads` (stage 3) attend, on every step, the blocks holding an earlier
occurrence of the current token or the token right after one (duplicate-token and
induction targets, found by token-id lookup without K) + sink + local blocks, and
never refresh.

With token ids available (STATE.ids) every record also carries the token-index
diagnostics: block mass covered by that index set (min over the G heads), the
token-level mass on its positions, and the mass on token classes (same token as the
current one, right after one, special tokens, newline tokens, sink block, local blocks).

Online calibration (stage 4, `calib` = its rate): on a refresh step a head reads every
block, so it learns the mass its stale set really missed, next to the pre-pass estimate it
decided with. Each KV head keeps a correction c and decides on the corrected estimate:
  logit  c = EMA of logit(real miss) - logit(estimated miss), decision sigmoid(logit(est) + c)
  add    c = EMA of real - estimated, decision est + c
  peak   c = max((1 - calib) c, real - estimated): a decaying maximum, so a single large
         invisible miss keeps the head sensitive for ~1/calib samples; decision est + c
(real and estimated misses are the max over the G query heads). By default c >= 0: the
correction can only raise an estimate. `calib_hold` delays applying it,
`calib_warm` = (W, A) refreshes every head at least every A steps during the first W steps
(more samples while c is cold), and `calib_auto_ix` latches a head onto the token index
(index union) once c exceeds that threshold.

Baselines at a byte budget (stage 4): "quest" reads, on every step, the blocks with the
highest Quest bound (per-channel key min/max of `quest_sub`-token sub-blocks, loaded by 64;
query heads ranked on their own normalised bound, max over the group) up to `budget` of the
blocks; "static" keeps the top `budget` blocks of the first decode step's exact attention
(GQA union of per-head ranks) and never updates them (SnapKV/H2O-like). Both always read
the sink block and the `local` newest blocks, inside the budget.

Every step and layer records per KV head: mass covered by the blocks actually
used (min / mean over the G heads), the mass the stale set would have covered,
fraction of blocks used, refresh flag, query drift and age of the set.

A2 -- run lengths of union top-p 0.99 masks at block sizes 16/32/64/128 and
cheaper block rankings (Quest min/max at 64 and at 16/32-token sub-blocks,
key centroid), accumulated on the exact reference pass.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import metrics
from metrics import BLOCK


@dataclass
class ReuseConfig:
    mode: str                         # fresh | timer | oracle | drift | estc | estd | vtau | vsink | vprev | vprev2
    N: int | None = None              # timer period
    theta: float | None = None        # oracle: refresh if stale coverage < theta
    delta: float | None = None        # drift: refresh if 1 - cos > delta; estc/estd: if estimated miss > delta
    max_age: int | None = None        # refresh anyway after this many steps
    level: float = 0.99               # top-p mass of the stored set
    local: int = 2                    # newest blocks always attended
    refresh: str = "exact"            # exact | masked
    dense_layers: tuple = (0,)
    dense_heads: tuple = ()           # (layer, kv_head) pairs that refresh (= exact) on every step
    index_heads: tuple = ()           # (layer, kv_head) pairs on the token index (stage 3)
    index_kind: str = "token"         # token | bigram: the index the index heads use
    index_union: bool = False         # index heads attend stored set + index blocks and keep the trigger
    diag: bool = True                 # compute the miss estimates even when the policy does not use them
    calib: float | None = None        # stage 4: rate of the per-KV-head estimate correction (EMA rate / peak decay)
    calib_hold: int = 0               # steps before the correction is applied (it is learned from step 1)
    calib_warm: tuple = ()            # (W, A): max age A instead of max_age during the first W steps
    calib_free: bool = False          # allow negative corrections (default: c >= 0)
    calib_auto_ix: float | None = None  # log-odds threshold: a head above it joins the token index (latched)
    calib_kind: str = "logit"         # logit: c corrects log-odds (a ratio); add: c is added to the miss share;
                                      # peak: added too, but c = max((1 - calib) c, residual) holds the worst one
    budget: float | None = None       # quest / static / topk: K+V bytes (blocks + metadata) vs dense, all layers
    match: str | None = None          # quest / static / topk: budget = the bytes of this condition (same run)
    quest_sub: int = 16               # quest: sub-block size of the key min/max
    quest_union: str = "norm"         # quest, GQA: norm = max over the group of each head's bound minus its best
                                      # block's; raw = max of the raw bounds; soft = sum over the group of
                                      # softmax(bound); rr = round robin over the heads' rankings
    quest_native: bool = False        # quest: select and read quest_sub-token pages (Quest's own granularity)
    block_frac: float | None = None   # set by the runner from the budget: blocks read per sparse layer and step
    budget_target: float | None = None

    def name(self) -> str:
        if self.mode == "fresh":
            return "fresh"
        if self.mode in VSKIP_MODES:
            return f"{self.mode}_tau{self.delta:g}"
        if self.mode in BUDGET_MODES:
            tag = self.mode + (f"{self.quest_sub}" if self.mode == "quest" else "")
            if self.mode == "quest" and self.quest_union != "norm":
                tag += self.quest_union
            if self.mode == "quest" and self.quest_native:
                tag += "native"
            tag += f"_match-{self.match}" if self.match else f"_b{self.budget:g}"
            return tag + ("" if self.dense_layers == (0,) else "_dl" + "".join(map(str, self.dense_layers)))
        if self.mode == "timer":
            tag = f"timer_N{self.N}"
        elif self.mode == "oracle":
            tag = f"oracle_theta{self.theta:g}"
        else:
            tag = f"{self.mode}_delta{self.delta:g}"
        if self.max_age:
            tag += f"_age{self.max_age}"
        if self.refresh != "exact":
            tag += f"_{self.refresh}"
        if self.dense_heads:
            tag += f"_dh{len(self.dense_heads)}"
        if self.index_heads:
            tag += (f"_ix{'u' if self.index_union else ''}{len(self.index_heads)}"
                    + ("b" if self.index_kind == "bigram" else ""))
        if self.calib is not None:
            tag += f"_cal{self.calib:g}"
            if self.calib_hold:
                tag += f"h{self.calib_hold}"
            if self.calib_warm:
                tag += f"w{self.calib_warm[0]}a{self.calib_warm[1]}"
            if self.calib_free:
                tag += "free"
            if self.calib_kind != "logit":
                tag += self.calib_kind
            if self.calib_auto_ix is not None:
                tag += f"_aix{self.calib_auto_ix:g}"
        if self.level != 0.99:
            tag += f"_p{self.level:g}"
        if self.dense_layers != (0,):
            tag += "_dl" + ("".join(map(str, self.dense_layers)) or "none")
        return tag


VSKIP_MODES = ("vtau", "vsink", "vprev", "vprev2")
BUDGET_MODES = ("quest", "static", "topk")     # stage 4 baselines at a byte budget (topk: per-step oracle)


def top_blocks(score: torch.Tensor, frac: float, always_row: torch.Tensor) -> torch.Tensor:
    """The round(frac * nb) highest-scoring blocks per row of score [Hkv, nb], the sink and
    local blocks always among them -> bool [Hkv, nb]."""
    nb = score.shape[-1]
    n = min(nb, max(int(always_row.sum()), round(frac * nb)))
    top = score.masked_fill(always_row[None, :], float("inf")).topk(n, dim=-1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter(-1, top, True)


def union_scores(ub: torch.Tensor, how: str) -> torch.Tensor:
    """Per-query-head bounds [Hkv, G, n] -> one score per KV head [Hkv, n] (see ReuseConfig.quest_union)."""
    if how == "norm":
        return (ub - ub.amax(-1, keepdim=True)).amax(1)
    if how == "raw":
        return ub.amax(1)
    if how == "soft":
        return torch.softmax(ub, dim=-1).sum(1)
    if how == "rr":                                   # round robin: a block's best rank over the group
        rank = ub.argsort(-1, descending=True).argsort(-1).float()          # 0 = best, per head
        G = ub.shape[1]
        key = rank * G + torch.arange(G, device=ub.device, dtype=torch.float32)[None, :, None]
        return -key.amin(1)
    raise ValueError(how)


def quest_bounds(q, k, scaling: float, sub: int) -> torch.Tensor:
    """Quest bound of every `sub`-token sub-block per query head: sum_d max(q_d min_d, q_d max_d)
    (per-channel key min / max) -> [Hkv, G, nsub], -inf past the end."""
    T = k.shape[1]
    kt = k.transpose(1, 2)                                                   # [Hkv, D, T]
    kmax = blocks_of(kt, sub, float("-inf")).amax(-1).transpose(1, 2)       # [Hkv, nsub, D]
    kmin = blocks_of(kt, sub, float("inf")).amin(-1).transpose(1, 2)
    nsub = kmax.shape[1]
    valid = torch.arange(nsub, device=k.device) * sub < T
    kmax, kmin = kmax.masked_fill(~valid[None, :, None], 0.0), kmin.masked_fill(~valid[None, :, None], 0.0)
    qe = q.unsqueeze(2)
    ub = torch.maximum(qe * kmin[:, None], qe * kmax[:, None]).sum(-1) * scaling
    return ub.masked_fill(~valid, float("-inf"))


def quest_scores(q, k, scaling: float, sub: int, how: str = "norm") -> torch.Tensor:
    """Quest bound of each 64-block: the max over its sub-blocks of sum_d max(q_d min_d, q_d max_d)
    (per-channel key min/max of `sub`-token sub-blocks), combined over the G heads of a KV head
    by `how` (default: each head's bound minus its best block's, max over the group) -> [Hkv, nb]."""
    if how != "norm":
        ub = blocks_of(quest_bounds(q, k, scaling, sub), BLOCK // sub, float("-inf")).amax(-1)
        return union_scores(ub, how)
    T = k.shape[1]
    kt = k.transpose(1, 2)                                                   # [Hkv, D, T]
    kmax = blocks_of(kt, sub, float("-inf")).amax(-1).transpose(1, 2)       # [Hkv, nsub, D]
    kmin = blocks_of(kt, sub, float("inf")).amin(-1).transpose(1, 2)
    nsub = kmax.shape[1]
    valid = torch.arange(nsub, device=k.device) * sub < T                    # sub-blocks holding a token
    kmax, kmin = kmax.masked_fill(~valid[None, :, None], 0.0), kmin.masked_fill(~valid[None, :, None], 0.0)
    qe = q.unsqueeze(2)                                                      # [Hkv, G, 1, D]
    ub = torch.maximum(qe * kmin[:, None], qe * kmax[:, None]).sum(-1) * scaling   # [Hkv, G, nsub]
    ub = ub.masked_fill(~valid, float("-inf"))
    ub = blocks_of(ub, BLOCK // sub, float("-inf")).amax(-1)                 # [Hkv, G, nb]
    return (ub - ub.amax(-1, keepdim=True)).amax(1)


def quest_native(cfg, records, s, p, q, k, scaling: float, layer: int, always_row, first: bool):
    """Quest at its own granularity: the top block_frac of the quest_sub-token pages by bound (sink
    and local 64-blocks always in), read token-exactly; `kept` is the share of pages read."""
    Hkv, G, T = s.shape
    dev = s.device
    sub = cfg.quest_sub
    score = union_scores(quest_bounds(q, k, scaling, sub), cfg.quest_union)          # [Hkv, nsub]
    nsub = score.shape[-1]
    always_sub = always_row.repeat_interleave(BLOCK // sub)[:nsub]
    sel = top_blocks(score, cfg.block_frac, always_sub)                               # [Hkv, nsub]
    tok = sel.repeat_interleave(sub, dim=-1)[:, :T]                                   # [Hkv, T]
    covered = (p * tok[:, None, :]).sum(-1)                                           # [Hkv, G]
    zeros = torch.zeros(Hkv, device=dev)
    records[layer] = dict(covered_min=covered.amin(-1), covered_mean=covered.mean(-1),
                          covered_stale_min=covered.amin(-1), kept=sel.float().mean(-1),
                          refresh=torch.full((Hkv,), first, dtype=torch.bool, device=dev),
                          drift=zeros, age=zeros, miss_c=zeros, miss_d=zeros, calib=zeros, auto_ix=zeros)
    if bool(tok.all()):
        return p
    return torch.softmax(s.masked_fill(~tok[:, None, :], float("-inf")), dim=-1)


def budgeted(cfg, states, records, s, p, q, k, scaling: float, layer: int, always_row):
    """quest / static / topk at a fixed share cfg.block_frac of the blocks per KV head and step."""
    Hkv, G, T = s.shape
    dev = s.device
    W = metrics.block_weights(p)                                             # [Hkv, G, nb]
    zeros = torch.zeros(Hkv, device=dev)
    first = layer not in states
    states.setdefault(layer, {})
    if cfg.mode == "quest" and cfg.quest_native:          # Quest's own pages: select and read sub-blocks
        return quest_native(cfg, records, s, p, q, k, scaling, layer, always_row, first)
    if cfg.mode == "quest":
        sel = top_blocks(quest_scores(q, k, scaling, cfg.quest_sub, cfg.quest_union), cfg.block_frac, always_row)
    elif cfg.mode == "topk":                                                 # oracle: this step's exact weights
        sel = top_blocks(W.amax(1), cfg.block_frac, always_row)
    elif first:                                                              # static: the first step's weights
        sel = top_blocks(W.amax(1), cfg.block_frac, always_row)
        states[layer]["sel"] = sel
    else:
        stored = states[layer]["sel"]
        sel = F.pad(stored, (0, W.shape[-1] - stored.shape[-1]), value=False) | always_row[None, :]
    if cfg.mode == "static" and first:
        sel = torch.ones_like(sel)                                           # the first step itself is exact
    covered = (W * sel[:, None, :]).sum(-1)                                  # [Hkv, G]
    refresh = torch.full((Hkv,), first, dtype=torch.bool, device=dev)
    records[layer] = dict(covered_min=covered.amin(-1), covered_mean=covered.mean(-1),
                          covered_stale_min=covered.amin(-1), kept=sel.float().mean(-1), refresh=refresh,
                          drift=zeros, age=zeros, miss_c=zeros, miss_d=zeros, calib=zeros, auto_ix=zeros)
    if bool(sel.all()):
        return p
    drop = (~sel).repeat_interleave(BLOCK, dim=-1)[:, :T]
    return torch.softmax(s.masked_fill(drop[:, None, :], float("-inf")), dim=-1)


def vskip(cfg: ReuseConfig, states: dict, records: dict, s, p, layer: int):
    """Single-pass V-skip rules (K is always read in full; the decision for a
    block's V is taken after its QK, per GQA union, with threshold tau = cfg.delta):
      vtau   exact block weight < tau                         (oracle, = stage 1 "weight" rule)
      vsink  exp(lse_b - lse_sink) < tau: the sink block's log-sum-exp is a lower
             bound of the normaliser that every CTA of a split-K kernel can compute
             by reading block 0 first, so the rule never drops a block with W >= tau
      vprev  exp(lse_b - lse_prev) < tau: the previous step's exact normaliser
      vprev2 the same with a 2-nat safety margin
    Block 0 and the `local` newest blocks are always kept."""
    Hkv, G, T = s.shape
    nb = math.ceil(T / BLOCK)
    dev = s.device
    lse_b = block_lse(s)                                           # [Hkv, G, nb]
    lse = s.logsumexp(-1)                                          # [Hkv, G]
    key = ("lse_prev", layer)
    prev = states.get(key)
    states[key] = lse
    if cfg.mode == "vtau":
        skip = lse_b - lse[..., None] < math.log(cfg.delta)
    elif cfg.mode == "vsink":
        skip = lse_b - lse_b[..., :1] < math.log(cfg.delta)
    else:
        if prev is None:                                           # first step: no estimate yet -> dense
            skip = torch.zeros(Hkv, G, nb, dtype=torch.bool, device=dev)
        else:
            margin = 2.0 if cfg.mode == "vprev2" else 0.0
            skip = lse_b - prev[..., None] < math.log(cfg.delta) - margin
    skip = skip.all(1)                                             # union over the G heads
    skip[:, 0] = False
    skip[:, -cfg.local:] = False
    keep = ~skip
    W = metrics.block_weights(p)
    covered = (W * keep[:, None, :]).sum(-1)
    zeros = torch.zeros(Hkv, device=dev)
    dlse = (lse - prev).abs().amax(-1) if prev is not None else zeros
    records[layer] = dict(covered_min=covered.amin(-1), covered_mean=covered.mean(-1),
                          covered_stale_min=covered.amin(-1), kept=keep.float().mean(-1),
                          refresh=torch.ones(Hkv, dtype=torch.bool, device=dev), drift=dlse,
                          age=zeros, miss_c=zeros, miss_d=zeros)
    if bool(keep.all()):
        return p
    drop = skip.repeat_interleave(BLOCK, dim=-1)[:, :T]
    return torch.softmax(s.masked_fill(drop[:, None, :], float("-inf")), dim=-1)


def block_lse(s: torch.Tensor) -> torch.Tensor:
    """log sum exp of the scores in each 64-block: [..., nb]."""
    return blocks_of(s, BLOCK, float("-inf")).logsumexp(-1)


def centroid_scores(q, k, scaling):
    """q [Hkv, G, D], k [Hkv, T, D] -> (q . block key centroid * scale [Hkv, G, nb], log n_b [nb])."""
    T = k.shape[1]
    c = blocks_of(k.transpose(1, 2), BLOCK, 0.0).sum(-1).transpose(1, 2)   # [Hkv, nb, D]
    nb = c.shape[1]
    sizes = torch.full((nb,), float(BLOCK), device=k.device)
    sizes[-1] = T - BLOCK * (nb - 1)
    c = c / sizes[None, :, None]
    return torch.einsum("kgd,knd->kgn", q, c) * scaling, sizes.log()


def est_miss(est_lse, attended_lse, unsel):
    """Estimated share of the mass outside the attended set, max over the G heads -> [Hkv].

    est_lse [Hkv, G, nb] per-block estimates, attended_lse [Hkv, G] exact, unsel [Hkv, nb]."""
    out = est_lse.masked_fill(~unsel[:, None, :], float("-inf")).logsumexp(-1)   # [Hkv, G]
    return torch.sigmoid(out - attended_lse).amax(-1)


CALIB_EPS = 1e-6                      # miss shares are clamped to [eps, 1 - eps] before the logit


def logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(CALIB_EPS, 1.0 - CALIB_EPS)
    return torch.log(x) - torch.log1p(-x)


def calib_update(cfg, calib, refresh, real, est):
    """EMA of logit(real) - logit(est) (kind "add": of real - est) on the heads that refresh
    (they read every block, so they know the real miss of their stale set); c >= 0 unless
    cfg.calib_free."""
    resid = logit(real) - logit(est) if cfg.calib_kind == "logit" else real - est
    if cfg.calib_kind == "peak":                                   # a decaying maximum: the tail, not the mean
        new = torch.maximum((1.0 - cfg.calib) * calib, resid)
    else:
        new = (1.0 - cfg.calib) * calib + cfg.calib * resid
    if not cfg.calib_free:
        new = new.clamp_min(0.0)
    return torch.where(refresh, new, calib)


def calib_apply(cfg, est, calib):
    return torch.sigmoid(logit(est) + calib) if cfg.calib_kind == "logit" else est + calib   # add, peak


DIAG_KEYS = ("miss_p", "idx_cov", "idx_tok", "idx_kept", "idx2_cov", "idx2_tok", "idx2_kept", "m_same", "m_after",
             "m_special", "m_newline", "m_sink", "m_local")


def token_index(ids: torch.Tensor, T: int, bigram: bool = False):
    """Positions an index on token ids finds for the current token ids[T-1]: earlier
    occurrences (duplicate-token target) and the positions right after them (induction
    target); with bigram=True only occurrences preceded by the previous token ids[T-2]
    too (an index on token pairs). Returns (same [T], after [T]) bool masks."""
    x = ids[T - 1]
    same = torch.zeros(T, dtype=torch.bool, device=ids.device)
    same[:T - 1] = ids[:T - 1] == x
    if bigram:
        prev = torch.zeros_like(same)
        prev[1:T - 1] = ids[:T - 2] == ids[T - 2]
        same &= prev
    after = torch.zeros_like(same)
    after[1:] = same[:T - 1]
    return same, after


def index_blocks(pos: torch.Tensor, always_row: torch.Tensor) -> torch.Tensor:
    return (blocks_of(pos.float(), BLOCK, 0.0).amax(-1) > 0) | always_row


def index_diag(p, W, ids, vocab, always_row, local: int):
    """Token-index diagnostics for one layer: p [Hkv, G, T], W [Hkv, G, nb].
    Returns (diagnostics, token-index blocks [nb], bigram-index blocks [nb])."""
    Hkv, G, T = p.shape
    same, after = token_index(ids, T)
    same2, after2 = token_index(ids, T, bigram=True)
    pos, pos2 = same | after, same2 | after2
    idx, idx2 = index_blocks(pos, always_row), index_blocks(pos2, always_row)
    m = lambda mask: (p * mask).sum(-1).mean(-1)                                # [Hkv], mean over G
    d = dict(idx_cov=(W * idx).sum(-1).amin(-1), idx_tok=m(pos), m_same=m(same), m_after=m(after),
             idx_kept=idx.float().mean().expand(Hkv),
             idx2_cov=(W * idx2).sum(-1).amin(-1), idx2_tok=m(pos2), idx2_kept=idx2.float().mean().expand(Hkv),
             m_sink=W[..., 0].mean(-1), m_local=W[..., -local:].sum(-1).mean(-1))
    if vocab is not None:
        tok = ids[:T]
        d["m_special"] = m(vocab["special"][tok])
        d["m_newline"] = m(vocab["newline"][tok])
    return d, idx, idx2


def apply(cfg: ReuseConfig, states: dict, records: dict, s, p, q, layer: int, step: int, k=None,
          scaling: float = 1.0, ids=None, vocab=None):
    """s, p [Hkv, G, T] exact fp32 scores/probs, q [Hkv, G, D], k [Hkv, T, D] fp32,
    ids [>= T] token ids of the sequence (optional: token-index diagnostics / index heads).
    Returns the probs to use."""
    Hkv, G, T = s.shape
    nb = math.ceil(T / BLOCK)
    dev = s.device
    ones = torch.ones(Hkv, device=dev)
    zeros = torch.zeros(Hkv, device=dev)
    diag = {key: zeros for key in DIAG_KEYS}
    idx = None
    W = metrics.block_weights(p)                                   # [Hkv, G, nb]
    always_row = torch.zeros(nb, dtype=torch.bool, device=dev)
    always_row[0] = True
    always_row[-cfg.local:] = True
    if ids is not None:
        d, idx, idx2 = index_diag(p, W, ids, vocab, always_row, cfg.local)
        diag.update(d)
        if cfg.index_kind == "bigram":
            idx = idx2
    if layer in cfg.dense_layers:
        records[layer] = dict(covered_min=ones, covered_mean=ones, covered_stale_min=ones, kept=ones,
                              refresh=torch.ones(Hkv, dtype=torch.bool, device=dev),
                              drift=zeros, age=zeros, miss_c=zeros, miss_d=zeros, calib=zeros, auto_ix=zeros,
                              **diag)
        return p
    if cfg.mode in VSKIP_MODES:
        out = vskip(cfg, states, records, s, p, layer)
        records[layer].update(diag, calib=zeros, auto_ix=zeros)
        return out
    if cfg.mode in BUDGET_MODES:
        out = budgeted(cfg, states, records, s, p, q, k, scaling, layer, always_row)
        records[layer].update(diag)
        return out

    fresh = metrics.needed_blocks(W, cfg.level).any(1)            # [Hkv, nb] union top-p
    always = always_row.expand(Hkv, nb)
    ix = [h for (l, h) in cfg.index_heads if l == layer]
    assert not ix or idx is not None, "index heads need token ids"
    extra = torch.zeros(Hkv, nb, dtype=torch.bool, device=dev)    # index blocks joining the set
    if ix and cfg.index_union:
        extra[ix] = idx
    qn = F.normalize(q, dim=-1)

    need_est = cfg.diag or cfg.mode in ("estc", "estd", "estp")
    if need_est:
        cs, log_n = centroid_scores(q, k, scaling)                 # [Hkv, G, nb], [nb]
        lse_now = block_lse(s)                                     # exact, used only on refresh
    else:
        cs = lse_now = torch.zeros(Hkv, G, nb, device=dev)

    st = states.get(layer)
    no_heads = torch.zeros(Hkv, dtype=torch.bool, device=dev)
    calib = zeros if st is None else st["calib"]                   # [Hkv] log-odds correction (stage 4)
    auto_ix = no_heads if st is None else st["auto_ix"]            # [Hkv] latched onto the token index
    if cfg.calib_auto_ix is not None:
        assert idx is not None, "calib_auto_ix needs token ids"
        extra = extra | (auto_ix[:, None] & idx[None, :])
    if st is None:
        refresh = torch.ones(Hkv, dtype=torch.bool, device=dev)
        stored = fresh
        age = torch.zeros(Hkv, dtype=torch.long, device=dev)
        drift, miss_c, miss_d, miss_p = zeros, zeros, zeros, zeros
        covered_stale = torch.ones(Hkv, device=dev)
        qn_ref = qn
        lse_ref, cs_ref = lse_now, cs
    else:
        stored = F.pad(st["sel"], (0, nb - st["sel"].shape[-1]), value=False)
        age = st["age"] + 1
        qn_ref = st["qn_ref"]
        drift = (1.0 - (qn * qn_ref).sum(-1)).amax(-1)                       # [Hkv]
        stale = stored | always | extra
        covered_stale = (W * stale[:, None, :]).sum(-1).amin(-1)            # [Hkv]
        # cheap estimates of the mass the stale set misses (blocks of the refresh
        # step only; newer blocks are always inside the local window)
        nbr = st["lse_ref"].shape[-1]
        if need_est:
            attended = s.masked_fill(~stale.repeat_interleave(BLOCK, -1)[:, None, :T],
                                     float("-inf")).logsumexp(-1)
            unsel = ~stale
            unsel[:, nbr:] = False
            miss_c = est_miss(log_n + cs, attended, unsel)
            est_d = F.pad(st["lse_ref"] + cs[..., :nbr] - st["cs_ref"], (0, nb - nbr), value=float("-inf"))
            miss_d = est_miss(est_d, attended, unsel)
            miss_p = est_miss(est_d, st["lse_att"], unsel)       # attended part from the previous step
        else:
            miss_c = miss_d = miss_p = zeros
        lse_ref, cs_ref = st["lse_ref"], st["cs_ref"]
        miss_dec = miss_p
        if cfg.calib is not None and step >= cfg.calib_hold:
            assert cfg.mode == "estp", "the calibration corrects the pre-pass estimate"
            miss_dec = calib_apply(cfg, miss_p, calib)
        if cfg.mode == "fresh":
            refresh = torch.ones(Hkv, dtype=torch.bool, device=dev)
        elif cfg.mode == "timer":
            refresh = torch.full((Hkv,), step % cfg.N == 0, dtype=torch.bool, device=dev)
        elif cfg.mode == "oracle":
            refresh = covered_stale < cfg.theta
        elif cfg.mode == "drift":
            refresh = drift > cfg.delta
        elif cfg.mode == "estc":
            refresh = miss_c > cfg.delta
        elif cfg.mode == "estd":
            refresh = miss_d > cfg.delta
        elif cfg.mode == "estp":
            refresh = miss_dec > cfg.delta
        else:
            raise ValueError(cfg.mode)
        if cfg.max_age:
            age_lim = cfg.max_age
            if cfg.calib_warm and step < cfg.calib_warm[0]:
                age_lim = min(age_lim, cfg.calib_warm[1])
            refresh = refresh | (age >= age_lim)
        forced = [h for (l, h) in cfg.dense_heads if l == layer]
        if forced:
            refresh = refresh.clone()
            refresh[forced] = True

    if cfg.mode == "fresh" or cfg.refresh == "masked":
        on_refresh = fresh | always
    else:
        on_refresh = torch.ones_like(fresh)
    sel_now = torch.where(refresh[:, None], on_refresh, stored | always | extra)
    if ix and not cfg.index_union:  # token-index heads: the index set alone, never a refresh
        refresh = refresh.clone()
        refresh[ix] = False
        sel_now = sel_now.clone()
        sel_now[ix] = idx

    r3 = refresh[:, None, None]
    if lse_ref.shape[-1] < nb:      # a new block appeared since the refresh step
        lse_ref = F.pad(lse_ref, (0, nb - lse_ref.shape[-1]), value=float("-inf"))
        cs_ref = F.pad(cs_ref, (0, nb - cs_ref.shape[-1]), value=0.0)
    sel_next = torch.where(refresh[:, None], fresh, stored)
    lse_att = lse_now.masked_fill(~(sel_next | always)[:, None, :], float("-inf")).logsumexp(-1)   # [Hkv, G]
    if cfg.calib is not None and st is not None:                  # learn from this step's refreshes
        calib = calib_update(cfg, calib, refresh, 1.0 - covered_stale, miss_p)
        if cfg.calib_auto_ix is not None:
            auto_ix = auto_ix | (calib > cfg.calib_auto_ix)
    states[layer] = dict(sel=sel_next, age=torch.where(refresh, torch.zeros_like(age), age),
                         qn_ref=torch.where(r3, qn, qn_ref), lse_att=lse_att,
                         lse_ref=torch.where(r3, lse_now, lse_ref), cs_ref=torch.where(r3, cs, cs_ref),
                         calib=calib, auto_ix=auto_ix)
    covered = (W * sel_now[:, None, :]).sum(-1)                   # [Hkv, G]
    diag["miss_p"] = miss_p
    records[layer] = dict(covered_min=covered.amin(-1), covered_mean=covered.mean(-1),
                          covered_stale_min=covered_stale, kept=sel_now.float().mean(-1),
                          refresh=refresh, drift=drift, age=age.float(), miss_c=miss_c, miss_d=miss_d,
                          calib=calib, auto_ix=auto_ix.float(), **diag)
    if bool(sel_now.all()):
        return p
    drop = (~sel_now).repeat_interleave(BLOCK, dim=-1)[:, :T]     # [Hkv, T]
    return torch.softmax(s.masked_fill(drop[:, None, :], float("-inf")), dim=-1)


# ----------------------------------------------------------------------------- A2

LOC_BLOCKS = (16, 32, 64, 128)
RUN_CAP = 512                         # run-length histograms are clipped here


def blocks_of(x: torch.Tensor, bs: int, fill: float) -> torch.Tensor:
    T = x.shape[-1]
    nb = math.ceil(T / bs)
    if nb * bs != T:
        x = F.pad(x, (0, nb * bs - T), value=fill)
    return x.reshape(*x.shape[:-1], nb, bs)


def run_lengths(mask: torch.Tensor):
    """mask [R, n] bool -> (lengths, values) of all maximal runs, row by row."""
    R, n = mask.shape
    change = torch.ones_like(mask)
    change[:, 1:] = mask[:, 1:] != mask[:, :-1]
    starts = change.flatten().nonzero().squeeze(1)
    ends = torch.cat([starts[1:], torch.tensor([R * n], device=mask.device)])
    return ends - starts, mask.flatten()[starts]


class Locality:
    """Accumulates A2 statistics over decode steps (per layer, per block size)."""

    def __init__(self, n_layers: int, device):
        self.L = n_layers
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=device)
        self.acc = {bs: dict(kept_blocks=z(n_layers), blocks=z(n_layers), kept_tokens=z(n_layers),
                             tokens=z(n_layers), kept_runs=z(n_layers), skip_runs=z(n_layers),
                             kept_in_long=z(n_layers), skip_in_long=z(n_layers), skipped=z(n_layers),
                             hist_kept=z(n_layers, RUN_CAP + 1), hist_skip=z(n_layers, RUN_CAP + 1))
                    for bs in LOC_BLOCKS}
        self.rank = {k: z(n_layers) for k in ("oracle", "blockmax", "quest64", "quest32", "quest16",
                                              "centroid", "count")}

    @torch.no_grad()
    def update(self, layer: int, s, p, q, k, scaling: float):
        Hkv, G, T = p.shape
        for bs, a in self.acc.items():
            W = blocks_of(p, bs, 0.0).sum(-1)                          # [Hkv, G, nb]
            sel = metrics.needed_blocks(W, 0.99).any(1)               # [Hkv, nb]
            nb = sel.shape[-1]
            sizes = torch.full((nb,), float(bs), device=p.device)
            sizes[-1] = T - bs * (nb - 1)
            lengths, values = run_lengths(sel)
            kl, sl = lengths[values], lengths[~values]
            a["kept_blocks"][layer] += sel.sum()
            a["blocks"][layer] += sel.numel()
            a["kept_tokens"][layer] += (sel.float() * sizes).sum()
            a["tokens"][layer] += Hkv * T
            a["kept_runs"][layer] += kl.numel()
            a["skip_runs"][layer] += sl.numel()
            a["kept_in_long"][layer] += kl[kl >= 4].sum()
            a["skip_in_long"][layer] += sl[sl >= 4].sum()
            a["skipped"][layer] += sl.sum()
            a["hist_kept"][layer] += torch.bincount(kl.clamp(max=RUN_CAP), minlength=RUN_CAP + 1)
            a["hist_skip"][layer] += torch.bincount(sl.clamp(max=RUN_CAP), minlength=RUN_CAP + 1)

        # Block rankings at block 64 without loading K: blocks needed for 0.99
        # when blocks are taken in the order of each predictor (per query head).
        W = metrics.block_weights(p)                                   # [Hkv, G, nb]
        kt = k.transpose(1, 2)                                         # [Hkv, D, T]
        qe = q.unsqueeze(2)                                            # [Hkv, G, 1, D]

        def quest(sub):
            kmax = blocks_of(kt, sub, float("-inf")).amax(-1).transpose(1, 2).unsqueeze(1)
            kmin = blocks_of(kt, sub, float("inf")).amin(-1).transpose(1, 2).unsqueeze(1)
            ub = torch.maximum(qe * kmin, qe * kmax).sum(-1) * scaling  # [Hkv, G, nsub]
            return blocks_of(ub, BLOCK // sub, float("-inf")).amax(-1)  # max over sub-blocks of a 64-block

        centroid = blocks_of(kt, BLOCK, 0.0).sum(-1).transpose(1, 2)     # [Hkv, nb, D] (sum; divide below)
        sizes = torch.full((W.shape[-1],), float(BLOCK), device=p.device)
        sizes[-1] = T - BLOCK * (W.shape[-1] - 1)
        centroid = centroid / sizes[None, :, None]
        cscore = torch.einsum("kgd,knd->kgn", q, centroid) * scaling
        preds = {"blockmax": metrics.block_max(s), "quest64": quest(64), "quest32": quest(32),
                 "quest16": quest(16), "centroid": cscore}
        self.rank["oracle"][layer] += metrics.needed_blocks(W, 0.99).sum(-1).float().mean()
        for name, key in preds.items():
            self.rank[name][layer] += metrics.blocks_to_mass_by(W, key, 0.99).mean()
        self.rank["count"][layer] += 1

    def to_numpy(self) -> dict:
        out = {}
        for bs, a in self.acc.items():
            for name, t in a.items():
                out[f"b{bs}_{name}"] = t.cpu().numpy()
        for name, t in self.rank.items():
            out[f"rank_{name}"] = t.cpu().numpy()
        return out
