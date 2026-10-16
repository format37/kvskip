"""Stage 5: is our Quest the published Quest? The selection of src/reuse.py `quest_paper_mask` (the
"questp" condition) against Quest's own evaluation code, on real decode-time q/k (data/kv_dumps).

The reference below is mit-han-lab/Quest, evaluation/quest_attention.py (commit 54bbfb0, 2024-11-22),
copied with only the module plumbing removed: `local_heavy_hitter_mask` verbatim, and the decode path of
`forward` from repeat_kv to the softmax. It runs in fp32 here (Quest's own run upcasts the bound to fp32
too; its attention scores stay fp16), on the same post-RoPE q and K our probe sees.

Checks, per dump (prose / code / dialog at 32k), layer (9, 21, 31; layers 0-1 are dense in both), captured
decode step (0, 1, 16, 64) and token budget (64 ... 16384 per query head):
  * the token mask each query head attends is identical (pages: count of differing selections);
  * the attention output over that mask is identical (max abs error, fp32);
  * the bound is an upper bound: no page's true max score exceeds it;
  * byte accounting: pages a GQA kernel reads per KV head (union over the group) vs per query head.

    .venv/bin/python bench/test_quest.py          # writes results/bench/quest_reference.{json,md}
"""

import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
import reuse  # noqa: E402

DUMPS = ROOT / "data" / "kv_dumps"
OUT = ROOT / "results" / "bench"
BUDGETS = (64, 256, 1024, 2048, 4096, 8192, 16384)
CHUNK = 16


# ------------------------------------------------- reference: mit-han-lab/Quest evaluation/quest_attention.py

def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def quest_reference(query_states, key_states, value_states, num_key_value_groups, head_dim, token_budget,
                    chunk_size):
    """The decode path of Quest's `forward` (layers >= 2) after RoPE and the cache update:
    query_states [1, H, 1, D], key/value_states [1, Hkv, T, D] -> (attn_output [1, H, 1, D], mask_bottom)."""
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        head_dim
    )

    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    postive_query = query_states * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // chunk_size,
        chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

    kv_seq_len = seq_length
    token_budget = min(kv_seq_len, token_budget)

    attn_weights_for_selection = quantized_weight

    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, chunk_size
        )  # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

    mask_bottom = torch.tril(mask_bottom, diagonal=kv_seq_len - 1)
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, mask_bottom


# ----------------------------------------------------------------------------------------------- checks

def load_dump(path):
    from safetensors import safe_open
    with safe_open(str(path), "pt", device="cuda") as f:
        meta = f.metadata()
        layers = json.loads(meta["layers"])
        data = {l: {n: f.get_tensor(f"L{l}.{n}") for n in ("q", "k", "v")} for l in layers}
    return json.loads(meta["T"]), data


@torch.no_grad()
def main():
    rows = []
    for path in sorted(DUMPS.glob("8b_*_32768.safetensors")):
        Ts, data = load_dump(path)
        for layer, d in data.items():
            if layer < 2:
                continue
            Hkv, _, D = d["k"].shape
            for s, T in enumerate(Ts):
                q = d["q"][s].float()                                     # [H, D]
                k, v = d["k"][:, :T].float(), d["v"][:, :T].float()      # [Hkv, T, D]
                G = q.shape[0] // Hkv
                scale = 1.0 / math.sqrt(D)
                # the bound is an upper bound of every page's max score
                ub = reuse.quest_bounds(q.view(Hkv, G, D), k, scale, CHUNK)              # [Hkv, G, nsub]
                sc = torch.einsum("kgd,ktd->kgt", q.view(Hkv, G, D), k) * scale
                page_max = reuse.blocks_of(sc, CHUNK, float("-inf")).amax(-1)
                bound_violation = float((page_max - ub).clamp_min(0).max())
                for B in BUDGETS:
                    o_ref, m_ref = quest_reference(q[None, :, None], k[None], v[None], G, D, B, CHUNK)
                    m_ref = m_ref[0, :, 0]                                               # [H, T]
                    m_ours = reuse.quest_paper_mask(q.view(Hkv, G, D), k, scale, B, CHUNK).reshape(-1, T)
                    p = torch.softmax(sc.masked_fill(~m_ours.view(Hkv, G, T), float("-inf")), dim=-1)
                    o_ours = torch.einsum("kgt,ktd->kgd", p, v).reshape(-1, D)
                    pages_ref = reuse.blocks_of(m_ref.float(), CHUNK, 0.0).amax(-1)       # [H, nsub]
                    pages_ours = reuse.blocks_of(m_ours.float(), CHUNK, 0.0).amax(-1)
                    nsub = pages_ref.shape[-1]
                    union = pages_ours.view(Hkv, G, nsub).amax(1)                         # [Hkv, nsub]
                    rows.append({"dump": path.stem, "layer": layer, "step": s, "T": T, "budget": B,
                                 "pages_per_head": int(pages_ours[0].sum()),
                                 "pages_differ": int((pages_ref != pages_ours).sum()),
                                 "pages_total": int(pages_ref.numel()),
                                 "out_max_abs_err": float((o_ref[0, :, 0] - o_ours).abs().max()),
                                 "out_scale": float(o_ref.abs().max()),
                                 "bound_violation": bound_violation,
                                 "head_share": float(pages_ours.mean()), "union_share": float(union.mean())})
    worst = max(r["pages_differ"] for r in rows)
    err = max(r["out_max_abs_err"] for r in rows)
    viol = max(r["bound_violation"] for r in rows)
    by_budget = {}
    for r in rows:
        b = by_budget.setdefault(r["budget"], {"head": [], "union": [], "differ": 0, "pages": 0})
        b["head"].append(r["head_share"])
        b["union"].append(r["union_share"])
        b["differ"] += r["pages_differ"]
        b["pages"] += r["pages_total"]
    summary = {"cases": len(rows), "pages_differ_max": worst, "out_max_abs_err": err, "bound_violation_max": viol,
               "passed": worst == 0 and err < 1e-4 and viol <= 1e-4,
               "by_budget": {B: {"pages_per_query_head": float(sum(b["head"]) / len(b["head"])),
                                 "pages_per_kv_head_union": float(sum(b["union"]) / len(b["union"])),
                                 "pages_differ": b["differ"], "pages_compared": b["pages"]}
                             for B, b in sorted(by_budget.items())}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "quest_reference.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    lines = ["| token budget per query head | pages read per query head | pages read per KV head (union of 4) | "
             "selections differing from Quest's code |", "|---|---|---|---|"]
    for B, b in summary["by_budget"].items():
        lines.append(f"| {B} | {b['pages_per_query_head']:.1%} | {b['pages_per_kv_head_union']:.1%} | "
                     f"{b['pages_differ']} of {b['pages_compared']} |")
    lines.append("")
    lines.append(f"{len(rows)} cases (3 documents x layers 9, 21, 31 x decode steps 0, 1, 16, 64 x {len(BUDGETS)} budgets), 32k: "
                 f"max |output difference| {err:.1e}, max bound violation {viol:.1e} -> "
                 f"{'PASS' if summary['passed'] else 'FAIL'}")
    (OUT / "quest_reference.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
