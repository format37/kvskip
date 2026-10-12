"""Stage 3: the decode continuation every stage-3 run teacher-forces.

For each document: exact prompt of 32768 - 128 tokens, then 128 greedy steps of the
exact fp32 probe (EOS suppressed). The tokens are the forced continuation at every
context length and batch size of stage 3: the layer-major lab (src/lab.py) cannot
generate freely (its decode positions are processed one layer at a time), and the
same answer keeps the runs comparable. At 64k the question is the same, so the
32k answer is a plausible continuation there too.

Output: data/docs/s3/forced.json {doc: {"tokens": [...], "text": str}}.
"""

import argparse
import json

import torch

import common
from prepare_s3 import DOCS
from probe_attn import STATE
from run_collect import greedy_token

OUT = common.ROOT / "data" / "docs" / "s3" / "forced.json"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--docs", nargs="+", default=list(DOCS))
    args = ap.parse_args()
    tok, model = common.load("8b")
    eos_ids = list(model.generation_config.eos_token_id)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args.docs:
        ids = common.exact_prompt(tok, common.load_named_doc(name), args.ctx - args.steps)
        cache = common.prefill(model, ids[:-1])
        STATE.reset()
        token, toks = ids[-1], []
        for _ in range(args.steps):
            token, _ = greedy_token(common.step(model, token, cache), eos_ids)
            toks.append(token)
        out[name] = {"ctx": args.ctx, "tokens": toks, "text": tok.decode(toks)}
        print(f"{name}: {tok.decode(toks)[:160]!r}", flush=True)
        del cache
        torch.cuda.empty_cache()
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
