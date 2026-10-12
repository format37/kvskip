"""Per-head attention statistics for one decode query against the whole KV.

All functions take fp32 tensors laid out as [Hkv, G, T] (scores / probs),
i.e. query heads grouped by the KV head they share (GQA). Query head h of
the model is (h // G, h % G) here, matching transformers' repeat_kv.

Block = BLOCK consecutive KV positions starting from position 0; the last
block may be partial.
"""

import math

import torch

BLOCK = 64
SINK_TOKENS = 4
TAUS = (1e-4, 1e-3, 1e-2)          # block-weight thresholds
MASS_LEVELS = (0.9, 0.99)          # "blocks needed to collect this mass"

REUSE_LAGS = (1, 16)               # re-use the 0.99-mass block set chosen this many steps ago
LOCAL_BLOCKS = 2                   # ... always together with the newest blocks

# 2D histogram of (pre-exp gap, log10 block weight), accumulated per layer.
GAP_EDGES = (0.0, 40.0, 160)       # nats: m - max_score_in_block
LOGW_EDGES = (-16.0, 0.0, 160)     # log10 of block weight


def pad_to_blocks(x: torch.Tensor, fill: float) -> torch.Tensor:
    T = x.shape[-1]
    nb = math.ceil(T / BLOCK)
    if nb * BLOCK != T:
        x = torch.nn.functional.pad(x, (0, nb * BLOCK - T), value=fill)
    return x.reshape(*x.shape[:-1], nb, BLOCK)


def block_weights(p: torch.Tensor) -> torch.Tensor:
    """Sum of softmax weight per KV block: [..., nb]."""
    return pad_to_blocks(p, 0.0).sum(-1)


def block_max(s: torch.Tensor) -> torch.Tensor:
    """Max raw score q.k*scale per KV block: [..., nb]."""
    return pad_to_blocks(s, float("-inf")).amax(-1)


def needed_blocks(W: torch.Tensor, level: float) -> torch.Tensor:
    """Bool mask [..., nb] of the smallest block set whose weight sums to >= level."""
    w_sorted, idx = W.sort(-1, descending=True)
    before = w_sorted.cumsum(-1) - w_sorted       # mass collected before this block
    take_sorted = before < level
    return torch.zeros_like(take_sorted).scatter(-1, idx, take_sorted)


def quest_bound(qv: torch.Tensor, k: torch.Tensor, scaling: float) -> torch.Tensor:
    """Upper bound of the block max score from per-block key min/max (Quest-style).

    qv [Hkv, G, D], k [Hkv, T, D] -> [Hkv, G, nb]. A kernel would keep the
    per-block min/max of K (2 vectors per block) and decide without loading K.
    """
    kt = k.transpose(1, 2)                                     # [Hkv, D, T]
    kmax = pad_to_blocks(kt, float("-inf")).amax(-1).transpose(1, 2).unsqueeze(1)   # [Hkv, 1, nb, D]
    kmin = pad_to_blocks(kt, float("inf")).amin(-1).transpose(1, 2).unsqueeze(1)
    qe = qv.unsqueeze(2)                                       # [Hkv, G, 1, D]
    return torch.maximum(qe * kmin, qe * kmax).sum(-1) * scaling


def blocks_to_mass_by(W: torch.Tensor, key: torch.Tensor, level: float) -> torch.Tensor:
    """Blocks needed to collect `level` of the weight when taken in order of decreasing `key`."""
    w = W.gather(-1, key.argsort(-1, descending=True))
    return ((w.cumsum(-1) - w) < level).sum(-1).float()


def preexp_skip(M: torch.Tensor, m: torch.Tensor, tau: float) -> torch.Tensor:
    """Pre-exp rule: skip a block if exp(max_score_in_block - global_max) < tau.

    Decidable from QK^T alone (no exp, no normaliser l). Since
    exp(M_b - m) / l <= W_b <= BLOCK * exp(M_b - m) / l and l >= 1, the rule
    is exact only up to the factor BLOCK / l.
    """
    return (m.unsqueeze(-1) - M) > math.log(1.0 / tau)


def spearman_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation along the last dim (no tie correction)."""
    rx = x.argsort(-1).argsort(-1).float()
    ry = y.argsort(-1).argsort(-1).float()
    rx = rx - rx.mean(-1, keepdim=True)
    ry = ry - ry.mean(-1, keepdim=True)
    return (rx * ry).sum(-1) / (rx.norm(dim=-1) * ry.norm(dim=-1)).clamp_min(1e-12)


def head_stats(s: torch.Tensor, p: torch.Tensor, qv: torch.Tensor, k: torch.Tensor, scaling: float,
               history=()) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    """Statistics of one decode step in one layer.

    s, p: [Hkv, G, T] fp32 scores (already scaled, masked positions = -inf) and probs.
    qv [Hkv, G, D], k [Hkv, T, D]: fp32 query / keys (for the key-bound predictor).
    history: this layer's 0.99-mass block sets of the previous steps (oldest first).
    Returns (per_query_head, per_kv_head, hist2d, needed_0.99) where the dicts
    map metric name -> tensor of shape [Hkv, G] / [Hkv], hist2d is the
    flattened (gap, log10 W) histogram over all blocks of all heads and
    needed_0.99 is this step's block set for the history.
    """
    Hkv, G, T = p.shape
    W = block_weights(p)                       # [Hkv, G, nb]
    M = block_max(s)                           # [Hkv, G, nb]
    m = s.amax(-1)                             # [Hkv, G]
    nb = W.shape[-1]

    plogp = torch.where(p > 0, p * torch.log2(p), torch.zeros_like(p))
    ent = -plogp.sum(-1)

    q = {
        "entropy_bits": ent,
        "entropy_norm": ent / math.log2(T),
        "sink_mass": p[..., :SINK_TOKENS].sum(-1),
        "local_mass": p[..., -BLOCK:].sum(-1),
        "top_block_mass": W.amax(-1),
        "max_prob": p.amax(-1),
    }
    for tau in TAUS:
        below = W < tau
        q[f"frac_blocks_lt_{tau:g}"] = below.float().mean(-1)
        # weight dropped if those blocks are skipped: per head, and when a block
        # is skipped only if all G heads of the KV group are below tau
        q[f"mass_lost_lt_{tau:g}"] = (W * below).sum(-1)
        q[f"union_mass_lost_lt_{tau:g}"] = (W * below.all(1, keepdim=True)).sum(-1)
    needed = {}
    for level in MASS_LEVELS:
        needed[level] = needed_blocks(W, level)
        q[f"blocks_to_{level:g}"] = needed[level].sum(-1).float()
        q[f"frac_blocks_to_{level:g}"] = q[f"blocks_to_{level:g}"] / nb

    # Pre-exp criterion: does the per-block max score predict block weight?
    gap = m.unsqueeze(-1) - M                  # [Hkv, G, nb], >= 0
    logW = torch.log10(W.clamp_min(1e-30))
    q["spearman_blockmax_vs_weight"] = spearman_rows(-gap, logW)
    for tau in TAUS:
        skip = preexp_skip(M, m, tau)
        q[f"preexp_frac_skip_{tau:g}"] = skip.float().mean(-1)
        q[f"preexp_mass_lost_{tau:g}"] = (W * skip).sum(-1)
        # blocks the pre-exp rule drops although their real weight is >= tau
        q[f"preexp_false_skip_{tau:g}"] = (skip & (W >= tau)).float().sum(-1)
        # blocks the pre-exp rule keeps although their real weight is < tau
        q[f"preexp_extra_keep_{tau:g}"] = (~skip & (W < tau)).float().sum(-1)

    # Extra: how many blocks are needed if they are taken in the order of a
    # cheaper predictor instead of the true weight -- block max score (needs K,
    # no exp) or the Quest-style key min/max bound (needs neither K nor exp).
    ub = quest_bound(qv, k, scaling)
    for level in MASS_LEVELS:
        q[f"blocks_to_{level:g}_by_blockmax"] = blocks_to_mass_by(W, M, level)
        q[f"blocks_to_{level:g}_by_quest"] = blocks_to_mass_by(W, ub, level)
    q["quest_slack_median"] = (ub - M).median(-1).values         # nats above the true block max

    # Extra: mass captured by re-using an old step's 0.99 block set (+ newest blocks).
    for lag in REUSE_LAGS:
        if len(history) >= lag:
            sel = torch.nn.functional.pad(history[-lag], (0, nb - history[-lag].shape[-1]), value=False)
            sel[..., -LOCAL_BLOCKS:] = True
            q[f"reuse_mass_lag{lag}"] = (W * sel).sum(-1)
            q[f"reuse_frac_blocks_lag{lag}"] = sel.float().mean(-1)
        else:
            q[f"reuse_mass_lag{lag}"] = torch.full_like(m, float("nan"))
            q[f"reuse_frac_blocks_lag{lag}"] = torch.full_like(m, float("nan"))

    # GQA union: a KV block must be loaded if any query head of the group needs it.
    kv = {}
    for level in MASS_LEVELS:
        union = needed[level].any(1)           # [Hkv, nb]
        kv[f"union_blocks_to_{level:g}"] = union.sum(-1).float()
        kv[f"union_frac_blocks_to_{level:g}"] = union.float().mean(-1)
        kv[f"mean_head_blocks_to_{level:g}"] = needed[level].sum(-1).float().mean(-1)
        kv[f"max_head_blocks_to_{level:g}"] = needed[level].sum(-1).float().amax(-1)
    for tau in TAUS:
        kv[f"union_frac_blocks_lt_{tau:g}"] = (W < tau).all(1).float().mean(-1)
        kv[f"union_preexp_frac_skip_{tau:g}"] = preexp_skip(M, m, tau).all(1).float().mean(-1)
    kv["nblocks"] = torch.full((Hkv,), float(nb), device=p.device)

    hist = hist2d(gap.flatten(), logW.flatten())
    return q, kv, hist, needed[0.99]


def hist2d(gap: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
    g0, g1, gn = GAP_EDGES
    w0, w1, wn = LOGW_EDGES
    gi = ((gap - g0) / (g1 - g0) * gn).floor().clamp(0, gn - 1).long()
    wi = ((logw - w0) / (w1 - w0) * wn).floor().clamp(0, wn - 1).long()
    return torch.bincount(gi * wn + wi, minlength=gn * wn)


def skip_mask(s: torch.Tensor, p: torch.Tensor, criterion: str, tau: float, scope: str) -> torch.Tensor:
    """Bool mask [Hkv, G, nb] of KV blocks to drop for the ablation.

    criterion: "weight" (oracle: real block weight < tau),
               "preexp" (exp(block max score - global max) < tau) or
               "mass"   (oracle top-p: keep the fewest blocks holding >= tau of the
                         weight, i.e. tau is a mass level such as 0.99).
    scope:     "head" (each query head decides alone) or
               "union" (a block is dropped only if all G heads of the KV group drop it).
    The block holding the global max score is never dropped, so a row can
    never end up empty.
    """
    M = block_max(s)
    if criterion == "weight":
        skip = block_weights(p) < tau
    elif criterion == "preexp":
        skip = preexp_skip(M, s.amax(-1), tau)
    elif criterion == "mass":
        skip = ~needed_blocks(block_weights(p), tau)
    else:
        raise ValueError(criterion)
    top = M.argmax(-1, keepdim=True)
    skip = skip.scatter(-1, top, False)
    if scope == "union":
        skip = skip.all(1, keepdim=True).expand_as(skip)
    elif scope != "head":
        raise ValueError(scope)
    return skip
