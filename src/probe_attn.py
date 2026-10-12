"""'probe' attention implementation for HF transformers.

Prefill (q_len > 1) is delegated to SDPA: it is not what we study and a
materialised 32k x 32k matrix would not fit anyway. Decode (q_len == 1) is
plain PyTorch with the softmax materialised in fp32 -- the ground truth and
the source of statistics:

  * STATE.collect = True   -> metrics.head_stats for every layer, kept on GPU
                              until the caller drains them with take_step().
  * STATE.mask = (criterion, tau, scope)
                           -> drop KV blocks before the softmax (ablation),
                              i.e. renormalise over the kept blocks exactly
                              like a block-sparse kernel that never loads them.
  * STATE.reuse = reuse.ReuseConfig(...)
                           -> stage 2 / A1: attend over a block set chosen on
                              an earlier (refresh) step; the driver sets
                              STATE.t to the decode step index.
  * STATE.locality = reuse.Locality(...)
                           -> stage 2 / A2: mask run lengths, block sizes and
                              cheap block rankings on the exact weights.

Usage: import probe_attn; probe_attn.register(); then load the model with
attn_implementation="probe" (or model.set_attn_implementation("probe")).
"""

from collections import deque

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

import metrics
import reuse
from metrics import BLOCK


class ProbeState:
    def __init__(self):
        self.collect = False
        self.mask = None            # None or (criterion, tau, scope)
        self.passthrough = None     # None, "sdpa" or "eager": stock HF bf16 decode (noise floor)
        self.reuse = None           # None or reuse.ReuseConfig (A1)
        self.locality = None        # None or reuse.Locality (A2)
        self.ids = None             # token ids of the sequence so far [T] (stage 3 token index), set by the driver
        self.vocab = None           # {"special": bool [V], "newline": bool [V]} token classes (stage 3)
        self.reset()

    def reset(self):
        self.q_stats = {}           # layer -> {metric: [Hkv, G]}
        self.kv_stats = {}          # layer -> {metric: [Hkv]}
        self.hist = {}              # layer -> accumulated 2D histogram (whole run)
        self.skip = {}              # layer -> (frac blocks dropped, mass dropped) [Hkv, G]
        self.needed = {}            # layer -> deque of the last steps' 0.99-mass block sets
        self.t = 0                  # decode step index (A1 timer), set by the driver
        self.reuse_state = {}       # layer -> stored block set, its age and query (A1)
        self.reuse_rec = {}         # layer -> this step's A1 record

    def take_step(self):
        """Return and clear the per-layer records of the last forward."""
        out = (self.q_stats, self.kv_stats, self.skip)
        self.q_stats, self.kv_stats, self.skip = {}, {}, {}
        return out


STATE = ProbeState()


def decode_attention(query, key, value, attention_mask, scaling, layer_idx):
    """query [1, H, 1, D], key/value [1, Hkv, T, D] -> fp32 attn_output [1, 1, H, D]."""
    _, H, _, D = query.shape
    Hkv, T = key.shape[1], key.shape[2]
    G = H // Hkv
    q = query.float().view(Hkv, G, D)
    k = key[0].float()
    v = value[0].float()
    s = torch.einsum("kgd,ktd->kgt", q, k) * scaling
    if attention_mask is not None:
        am = attention_mask[0, 0, -1, :T]
        s = s.masked_fill(~am, float("-inf")) if am.dtype == torch.bool else s + am.float()
    p = torch.softmax(s, dim=-1)

    if STATE.collect:
        history = STATE.needed.setdefault(layer_idx, deque(maxlen=max(metrics.REUSE_LAGS)))
        qs, kvs, hist, needed = metrics.head_stats(s, p, q, k, scaling, history)
        history.append(needed)
        STATE.q_stats[layer_idx] = qs
        STATE.kv_stats[layer_idx] = kvs
        acc = STATE.hist.get(layer_idx)
        STATE.hist[layer_idx] = hist if acc is None else acc + hist

    if STATE.mask is not None:
        criterion, tau, scope = STATE.mask
        skip = metrics.skip_mask(s, p, criterion, tau, scope)          # [Hkv, G, nb]
        STATE.skip[layer_idx] = (skip.float().mean(-1),
                                 (metrics.block_weights(p) * skip).sum(-1))
        drop = skip.repeat_interleave(BLOCK, dim=-1)[..., :T]
        p = torch.softmax(s.masked_fill(drop, float("-inf")), dim=-1)

    if STATE.locality is not None:
        STATE.locality.update(layer_idx, s, p, q, k, scaling)
    if STATE.reuse is not None:
        p = reuse.apply(STATE.reuse, STATE.reuse_state, STATE.reuse_rec, s, p, q, layer_idx, STATE.t,
                        k=k, scaling=scaling, ids=STATE.ids, vocab=STATE.vocab)

    o = torch.einsum("kgt,ktd->kgd", p, v)
    return o.reshape(1, 1, H, D)


def probe_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if query.shape[2] != 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    if STATE.passthrough == "sdpa":
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    if STATE.passthrough == "eager":
        return eager_attention_forward(module, query, key, value, attention_mask,
                                       scaling=scaling, dropout=dropout, **kwargs)
    assert query.shape[0] == 1, "probe decode path assumes batch size 1"
    out = decode_attention(query, key, value, attention_mask, scaling, module.layer_idx)
    return out.to(query.dtype), None


def register():
    AttentionInterface.register("probe", probe_attention_forward)
    AttentionMaskInterface.register("probe", sdpa_mask)
