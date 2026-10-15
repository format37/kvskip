"""Stage 2 / A3: perplexity check of "free" sparsity on a held-out continuation.

Teacher forcing over 128 steps (A1, stage 1) hides error accumulation through
the KV cache of the generated tokens. Here every genre's document is cut into
a context of `ctx` tokens and the next `n` tokens (default 2048) of the same
document; the continuation is teacher-forced one token at a time through the
probe decode path, so the KV of every continuation token is computed under the
condition being tested and errors can accumulate over n steps.

Token streams: prose / code = raw document text with BOS; dialog = the
concatenated conversations through the chat template.
Conditions: exact fp32 probe (reference), stock bf16 SDPA, "fresh" (GQA-union
top-p 0.99 every step = two-pass V-skip) and any reuse policy given with --cond
(src/run_reuse.parse_cond syntax).

Output: results/ppl/<model>_<genre>_<ctx>.{tokens.parquet,summary.json}

Input: data/docs/<genre>.json (src/prepare_data.py), models/<model>.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
from probe_attn import STATE
from run_reuse import parse_cond

OUT = common.ROOT / "results" / "ppl"
WINDOWS = ((0, 256), (256, 1024), (1024, 2048))


def token_stream(tok, genre: str) -> list[int]:
    doc = common.load_doc(genre)
    if genre == "dialog":
        flat = [m for c in doc["conversations"] for m in c["messages"]]
        return list(tok.apply_chat_template(flat, tokenize=True, return_dict=False,
                                            date_string=common.DATE_STRING))
    body = doc.get("joiner", "\n\n").join(doc["paragraphs"])
    return tok(body, add_special_tokens=True)["input_ids"]


@torch.inference_mode()
def score(model, cache, prefix_len, ids, ctx, n, cfg=None, passthrough=None):
    """NLL (nats) and argmax hits of ids[ctx : ctx + n], teacher-forced from ids[ctx - 1]."""
    extra = cache.get_seq_length() - prefix_len
    if extra > 0:
        cache.crop(-extra)
    STATE.reset()
    STATE.reuse, STATE.passthrough = cfg, passthrough
    target = torch.tensor(ids[ctx:ctx + n], device=model.device)
    nll, hit = torch.empty(n, device=model.device), torch.empty(n, dtype=torch.bool, device=model.device)
    for i in range(n):
        STATE.t = i
        logits = common.step(model, ids[ctx - 1 + i], cache)
        lp = torch.log_softmax(logits, -1)
        nll[i] = -lp[target[i]]
        hit[i] = logits.argmax() == target[i]
    STATE.reuse, STATE.passthrough = None, None
    return nll.cpu().numpy(), hit.cpu().numpy()


def run(model_name, genre, ctx, n, tok, model, conds):
    ids = token_stream(tok, genre)
    assert len(ids) >= ctx + n, f"{genre}: {len(ids)} tokens < {ctx + n}"
    prefix_len = ctx - 1
    cache = common.prefill(model, ids[:prefix_len])
    rows, summary = {}, {"model": model_name, "genre": genre, "ctx": ctx, "n": n, "conditions": {}}
    plan = [("exact_fp32", None, None), ("stock_sdpa_bf16", None, "sdpa")]
    for spec in conds:
        cfg = parse_cond(spec)
        cfg.diag = False
        plan.append((cfg.name(), cfg, None))
    for name, cfg, passthrough in plan:
        t0 = time.perf_counter()
        nll, hit = score(model, cache, prefix_len, ids, ctx, n, cfg, passthrough)
        rows[name] = nll
        rows[name + "_hit"] = hit
        row = {"nll": float(nll.mean()), "ppl": float(math.exp(nll.mean())), "top1": float(hit.mean()),
               "seconds": round(time.perf_counter() - t0, 1)}
        for a, b in WINDOWS:
            row[f"nll_{a}_{b}"] = float(nll[a:b].mean())
        if name != "exact_fp32":
            d = nll - rows["exact_fp32"]
            row["dnll"] = float(d.mean())
            row["dnll_se"] = float(d.std(ddof=1) / math.sqrt(len(d)))
            row["dppl_pct"] = 100.0 * (math.exp(nll.mean() - rows["exact_fp32"].mean()) - 1.0)
            for a, b in WINDOWS:
                row[f"dnll_{a}_{b}"] = float(d[a:b].mean())
        summary["conditions"][name] = row
        print(f"  {name:28s} ppl={row['ppl']:.4f} nll={row['nll']:.4f} "
              f"dnll={row.get('dnll', 0):+.5f}±{row.get('dnll_se', 0):.5f} top1={row['top1']:.3f} "
              f"({row['seconds']}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{model_name}_{genre}_{ctx}"
    pd.DataFrame(rows).assign(model=model_name, genre=genre, ctx=ctx).to_parquet(f"{stem}.tokens.parquet")
    Path(f"{stem}.summary.json").write_text(json.dumps(summary, indent=1))
    del cache
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=list(common.MODELS))
    ap.add_argument("--genre", nargs="+", default=list(common.GENRES))
    ap.add_argument("--ctx", nargs="+", type=int, default=[32768])
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--cond", nargs="*", default=["fresh"])
    args = ap.parse_args()
    tok, model = common.load(args.model)
    for ctx in args.ctx:
        for genre in args.genre:
            print(f"[ppl] {args.model} {genre} ctx={ctx} n={args.n}", flush=True)
            run(args.model, genre, ctx, args.n, tok, model, args.cond)


if __name__ == "__main__":
    main()
