"""Paper: task accuracy on RULER at 32k, with the question read through the sparse decode path.

Data: RULER (github.com/NVIDIA/RULER, commit c3f5e3b) synthetic tasks for Llama-3.1-8B-Instruct, made by
its scripts/data/prepare.py (--tokenizer_type hf --model_template_type meta-llama3 --max_seq_length 32768
--num_samples 100, seed 42) into data/raw/ruler-data/<task>/validation.jsonl (reproduce/06-paper.sh).
The prompt is input + answer_prefix, as RULER's call_api.py builds it, tokenized once. Everything before
the question is prefilled densely; the question, the assistant header and the answer prefix go through
the decode path token by token (as Quest's passkey evaluation feeds its last 400 characters), so the
sparse attention reads them. Then the run's own greedy answer until EOS or RULER's tokens_to_generate,
scored with RULER's metric (string_match_all; string_match_part for QA) after its postprocess_pred.

Per sample: one prefill (shared); the exact fp32 reference (probe) and stock bf16 SDPA, both answering
on their own; then every condition (src/run_reuse.py parse_cond; match= takes the bytes of an earlier
condition on the same sample):
  * the score of its own answer;
  * K+V bytes as in run_reuse (blocks + metadata, all layers), over its decode steps;
  * KL(ref || cond) on the question steps, where all runs share the prefix, in multiples of stock
    SDPA's KL on the same steps.
Samples are taken task by task in turn (sample 0 of every task, then sample 1, ...), so a run cut short
stays balanced across tasks.

Output: results/ruler/<model>_<ctx>[_<tag>].json, rewritten after every sample (--resume skips the
samples it already holds).
Input: models/<model>, data/raw/ruler-data/<task>/validation.jsonl.
"""

import argparse
import json
import re
import time

import numpy as np
import torch

import common
import run_reuse
from probe_attn import STATE
from run_reuse import REC_KEYS, bytes_of, kl, parse_cond, set_budget, vocab_classes

OUT = common.ROOT / "results" / "ruler"
DATA = common.ROOT / "data" / "raw" / "ruler-data"
# From RULER's scripts/data/synthetic/constants.py (tokens_to_generate) and
# scripts/eval/synthetic/constants.py (metric); the marker is where the question starts.
TASKS = {
    "niah_multikey_2": (128, "all", "\nWhat "),
    "niah_multivalue": (128, "all", "\nWhat "),
    "vt": (30, "all", "\nQuestion:"),
    "fwe": (50, "all", "\nQuestion:"),
    "qa_1": (32, "part", "\n\nAnswer the question based"),
}


def postprocess(pred: str) -> str:
    """RULER's postprocess_pred."""
    return re.sub(r"[\x00-\x1f]", "\n", pred.strip()).strip()


def score(pred: str, refs: list[str], metric: str) -> float:
    """RULER's string_match_all / string_match_part for one sample, in [0, 1]."""
    hits = [r.lower() in pred.lower() for r in refs]
    return float(sum(hits) / len(refs)) if metric == "all" else float(max(hits))


def split_ids(tok, prompt: str, marker: str) -> tuple[list[int], list[int]]:
    """The prompt tokenized once (it carries its own <|begin_of_text|>); the decode part starts at the
    first token that ends after the question's marker."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1, "expected one leading BOS"
    b = prompt.rfind(marker)
    assert b > 0, f"marker {marker!r} not in the prompt"
    cut = next(i for i, (_, e) in enumerate(offs) if e > b)
    return ids[:cut], ids[cut:]


def decode(model, cache, prefix_len, q_ids, eos_ids, max_new, cfg=None):
    """Feed the question tokens through the decode path, then the run's own greedy answer until EOS or
    max_new tokens. Returns (log-probs [steps, V], answer ids, records)."""
    extra = cache.get_seq_length() - prefix_len
    if extra > 0:
        cache.crop(-extra)
    STATE.reset()
    STATE.reuse = cfg
    L = model.config.num_hidden_layers
    n_q = len(q_ids)
    lps, answer, recs = [], [], []
    token = q_ids[0]
    for i in range(n_q + max_new - 1):
        STATE.t = i
        IDS[prefix_len + i] = token
        STATE.ids = IDS[:prefix_len + i + 1]
        logits = common.step(model, token, cache)
        lps.append(torch.log_softmax(logits, -1))
        if cfg is not None:
            recs.append({k: torch.stack([STATE.reuse_rec[l][k].float() for l in range(L)]) for k in REC_KEYS})
        if i < n_q - 1:
            token = q_ids[i + 1]
            continue
        pick = int(logits.argmax())
        if pick in eos_ids:
            break
        answer.append(pick)
        token = pick
    STATE.reuse = None
    return torch.stack(lps), answer, recs


def run(tok, model, configs, task, sample, row):
    global IDS
    max_new, metric, marker = TASKS[task]
    eos_ids = common.eos_ids(model)
    ids, q_ids = split_ids(tok, row["input"] + row["answer_prefix"], marker)
    prefix_len, n_q = len(ids), len(q_ids)
    IDS = torch.zeros(prefix_len + n_q + max_new, dtype=torch.long, device=model.device)
    IDS[:prefix_len] = torch.tensor(ids, device=model.device)
    cache = common.prefill(model, ids)
    L = model.config.num_hidden_layers
    D = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    G = model.config.num_attention_heads // model.config.num_key_value_heads
    refs = row["outputs"]

    def text(answer):
        return postprocess(tok.decode(answer))

    t0 = time.perf_counter()
    ref_lp, ref_ans, _ = decode(model, cache, prefix_len, q_ids, eos_ids, max_new)
    STATE.passthrough = "sdpa"
    noise_lp, noise_ans, _ = decode(model, cache, prefix_len, q_ids, eos_ids, max_new)
    STATE.passthrough = None
    noise_kl = kl(ref_lp[:n_q], noise_lp[:n_q]).cpu().numpy()
    out = {"task": task, "sample": sample, "outputs": refs, "prompt_tokens": prefix_len + n_q,
           "question_tokens": n_q, "noise_kl_q": float(noise_kl.mean()),
           "ref": {"answer": text(ref_ans), "score": score(text(ref_ans), refs, metric), "steps": len(ref_lp)},
           "sdpa": {"answer": text(noise_ans), "score": score(text(noise_ans), refs, metric),
                    "answer_same": noise_ans == ref_ans, "steps": len(noise_lp)},
           "conditions": {}}
    print(f"[ruler] {task} #{sample}: prompt {prefix_len + n_q} tokens ({n_q} decoded), gold {refs}; "
          f"ref {out['ref']['answer']!r} ({out['ref']['score']:.2f}), sdpa {out['sdpa']['score']:.2f}, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    spent = {}
    for cfg in configs:
        if cfg.mode in run_reuse.reuse.BUDGET_MODES:
            target = cfg.budget if cfg.match is None else spent[cfg.match]["bytes_total"]
            set_budget(cfg, target, L, D, G, n_q + max_new)
        name = cfg.name()
        t0 = time.perf_counter()
        lp, ans, recs = decode(model, cache, prefix_len, q_ids, eos_ids, max_new, cfg=cfg)
        k = kl(ref_lp[:n_q], lp[:n_q]).cpu().numpy()
        R = {key: torch.stack([r[key] for r in recs]).cpu().numpy() for key in REC_KEYS}
        kept = R["kept"]
        k_loaded = np.ones_like(kept) if cfg.mode in ("fresh",) + run_reuse.reuse.VSKIP_MODES else kept
        spent[name] = bytes_of(cfg, kept, k_loaded, L, D, G)
        sparse = [l for l in range(L) if l not in cfg.dense_layers]
        out["conditions"][name] = c = {
            "answer": text(ans), "score": score(text(ans), refs, metric), "answer_same": ans == ref_ans,
            **spent[name], "steps": len(lp), "kl_q_x_noise": float(k.mean() / noise_kl.mean()),
            "kl_q_max": float(k.max()), "refresh_rate": float(R["refresh"][1:, sparse].mean()),
            "seconds": round(time.perf_counter() - t0, 1)}
        print(f"  {name:60s} {c['score']:.2f} {c['answer'][:40]!r:44s} KLq x{c['kl_q_x_noise']:.2f} "
              f"bytes={c['bytes_total']:.3f} {c['seconds']}s", flush=True)
    del cache
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=list(common.MODELS))
    ap.add_argument("--ctx", type=int, default=32768, help="names the output; the data fixes the length")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--n", type=int, default=50, help="samples per task")
    ap.add_argument("--start", type=int, default=0, help="first sample of every task")
    ap.add_argument("--cond", nargs="*", default=[], help="conditions, see run_reuse.parse_cond")
    ap.add_argument("--tag", default="", help="output name suffix")
    ap.add_argument("--resume", action="store_true", help="keep the samples the output already holds")
    args = ap.parse_args()
    tok, model = common.load(args.model)
    STATE.vocab = vocab_classes(tok, model.device, model.config.vocab_size)
    path = (OUT / (f"{args.model}_{args.ctx}" + (f"_{args.tag}" if args.tag else ""))).with_suffix(".json")
    runs = json.loads(path.read_text())["runs"] if args.resume and path.exists() else []
    done = {(r["task"], r["sample"]) for r in runs}
    rows = {t: [json.loads(line) for line in open(DATA / t / "validation.jsonl")] for t in args.tasks}
    for i in range(args.start, args.start + args.n):
        for task in args.tasks:
            if (task, i) in done:
                continue
            configs = [parse_cond(c) for c in args.cond]
            runs.append(run(tok, model, configs, task, i, rows[task][i]))
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": args.model, "ctx": args.ctx, "conds": args.cond, "runs": runs},
                                       indent=1))
    for task in args.tasks:
        rs = [r for r in runs if r["task"] == task]
        print(f"\n{task}: {len(rs)} samples; ref {np.mean([r['ref']['score'] for r in rs]):.3f}, "
              f"sdpa {np.mean([r['sdpa']['score'] for r in rs]):.3f}")
        for n in rs[0]["conditions"]:
            cs = [r["conditions"][n] for r in rs]
            print(f"  {n:60s} score {np.mean([c['score'] for c in cs]):.3f}  "
                  f"bytes {np.mean([c['bytes_total'] for c in cs]):.3f}  "
                  f"KLq x{np.exp(np.mean(np.log([c['kl_q_x_noise'] for c in cs]))):.2f} (gmean)")


if __name__ == "__main__":
    main()
