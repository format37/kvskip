"""Stage 5 (phase 1): Quest in its own regime -- passkey retrieval as Quest's evaluation runs it.

Task and protocol follow mit-han-lab/Quest evaluation/passkey/passkey.py (from landmark-attention): a
task line, filler ("The grass is green. ..."), the line "The pass key is N. Remember it. N is the pass
key." at depth d of the filler, more filler, "What is the pass key? The pass key is"; raw text, no chat
template (Quest's own scripts/passkey.sh runs Llama-3.1-8B-Instruct this way). All but the last 400
characters are prefilled densely; those 400 characters (filler and the question) go through the decode
path token by token, so the sparse attention reads them; then 10 greedy tokens, and the first integer of
the answer is compared with N. Depths 10, 20, ..., 100 % as in Quest's script with 10 iterations; N is
drawn from 1..50000 with a seed per depth.

Per depth: one prefill (shared), the exact fp32 reference (probe, free answer), stock bf16 SDPA
teacher-forced on the reference's tokens (noise floor), then for every condition (src/run_reuse.py
parse_cond; questp:<tokens> is Quest as published):
  * a free run: question tokens forced, the answer its own -> retrieved or not;
  * KL(ref || cond) on every decode step (question + answer) on the reference's tokens: from the free
    run when its answer equals the reference's, else from a second, teacher-forced run;
  * K+V bytes as in run_reuse (blocks + metadata, all layers, over the decode steps).

Output: results/passkey/<model>_<ctx>[_<tag>].json; with --save-heads also .heads.parquet (per depth,
condition, step, layer and KV head: coverage, refresh, the correction c).
Input: models/<model>.
"""

import argparse
import json
import random
import re
import time

import numpy as np
import pandas as pd
import torch

import common
import run_reuse
from probe_attn import STATE
from run_reuse import REC_KEYS, bytes_of, kl, parse_cond, set_budget, vocab_classes

OUT = common.ROOT / "results" / "passkey"
N_ANSWER = 10
Q_CHARS = 400


def generate_prompt(n_garbage: int, depth_ratio: int, pass_key: int) -> str:
    """Quest's / landmark-attention's generate_prompt with a given depth (percent) and key."""
    n_garbage_prefix = int(n_garbage * depth_ratio / 100)
    n_garbage_suffix = n_garbage - n_garbage_prefix
    task_description = ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize "
                        "them. I will quiz you about the important information there.")
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 10000)
    assert len(garbage_inf) >= n_garbage
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [task_description, garbage_inf[:n_garbage_prefix], information_line, garbage_inf[:n_garbage_suffix],
             final_question]
    return "\n".join(lines)


def split_ids(tok, prompt: str) -> tuple[list[int], list[int]]:
    """Quest's test_model: the prompt minus its last Q_CHARS characters is the prefill (with BOS); those
    characters are tokenized on their own, BOS dropped, and fed one token at a time."""
    text, que = prompt[:-Q_CHARS], prompt[-Q_CHARS:]
    return tok(text).input_ids, tok(que).input_ids[1:]


def garbage_for(tok, target: int) -> int:
    """The longest filler (characters, multiples of 128 like Quest's length_step) whose prompt fits
    in `target` tokens."""
    lo, hi = 0, 4 * target
    while hi - lo > 128:
        mid = (lo + hi) // 2
        a, b = split_ids(tok, generate_prompt(mid, 50, 49999))
        if len(a) + len(b) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def first_int(text: str):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def decode(model, cache, prefix_len, q_ids, eos_ids, answer=None, cfg=None):
    """Feed the question tokens through the decode path, then N_ANSWER answer tokens: `answer` (teacher
    forcing) or the run's own greedy picks. Returns (log-probs [steps, V], greedy picks, records)."""
    extra = cache.get_seq_length() - prefix_len
    if extra > 0:
        cache.crop(-extra)
    STATE.reset()
    STATE.reuse = cfg
    L = model.config.num_hidden_layers
    steps = len(q_ids) + N_ANSWER - 1
    lps, picks, recs = [], [], []
    token = q_ids[0]
    for i in range(steps):
        STATE.t = i
        IDS[prefix_len + i] = token
        STATE.ids = IDS[:prefix_len + i + 1]
        logits = common.step(model, token, cache)
        lps.append(torch.log_softmax(logits, -1))
        if cfg is not None:
            recs.append({k: torch.stack([STATE.reuse_rec[l][k].float() for l in range(L)]) for k in REC_KEYS})
        picks.append(int(logits.argmax()))
        j = i + 1 - len(q_ids)                       # answer index of the next token to feed
        token = q_ids[i + 1] if j < 0 else (answer[j] if answer is not None else picks[-1])
    STATE.reuse = None
    return torch.stack(lps), picks, recs


def answer_text(tok, picks, n_q, eos_ids) -> str:
    out = []
    for t in picks[n_q - 1:]:
        if t in eos_ids:
            break
        out.append(t)
    return tok.decode(out)


def run(tok, model, configs, ctx, depth, seed, n_garbage, head_rows=None):
    global IDS
    eos_ids = common.eos_ids(model)
    pass_key = random.Random(seed * 1000 + depth).randint(1, 50000)
    ids, q_ids = split_ids(tok, generate_prompt(n_garbage, depth, pass_key))
    prefix_len = len(ids)
    IDS = torch.zeros(prefix_len + len(q_ids) + N_ANSWER, dtype=torch.long, device=model.device)
    IDS[:prefix_len] = torch.tensor(ids, device=model.device)
    cache = common.prefill(model, ids)
    L = model.config.num_hidden_layers
    D = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    G = model.config.num_attention_heads // model.config.num_key_value_heads
    n_q = len(q_ids)

    t0 = time.perf_counter()
    ref_lp, ref_picks, _ = decode(model, cache, prefix_len, q_ids, eos_ids)
    ref_answer = ref_picks[n_q - 1:]
    ref_text = answer_text(tok, ref_picks, n_q, eos_ids)
    STATE.passthrough = "sdpa"
    noise_lp, noise_picks, _ = decode(model, cache, prefix_len, q_ids, eos_ids, answer=ref_answer)
    STATE.passthrough = None
    noise_kl = kl(ref_lp, noise_lp).cpu().numpy()
    out = {"depth": depth, "pass_key": pass_key, "prompt_tokens": prefix_len + n_q, "decode_steps": len(noise_kl),
           "question_tokens": n_q, "ref_answer": ref_text, "ref_retrieved": first_int(ref_text) == pass_key,
           "noise_kl_mean": float(noise_kl.mean()), "noise_answer_same": noise_picks[n_q - 1:] == ref_answer,
           "conditions": {}}
    print(f"[passkey] depth {depth}% key {pass_key}: prompt {prefix_len + n_q} tokens ({n_q} decoded), "
          f"ref {ref_text!r} ({'ok' if out['ref_retrieved'] else 'MISS'}) {time.perf_counter() - t0:.0f}s, "
          f"noise KL {noise_kl.mean():.2e}", flush=True)

    spent = {}
    for cfg in configs:
        if cfg.mode in run_reuse.reuse.BUDGET_MODES:
            target = cfg.budget if cfg.match is None else spent[cfg.match]["bytes_total"]
            set_budget(cfg, target, L, D, G, len(noise_kl))
        name = cfg.name()
        t0 = time.perf_counter()
        lp, picks, recs = decode(model, cache, prefix_len, q_ids, eos_ids, cfg=cfg)
        text = answer_text(tok, picks, n_q, eos_ids)
        forced = picks[n_q - 1:] != ref_answer
        if forced:                                   # KL on the reference's tokens needs a teacher-forced run
            lp, _, recs = decode(model, cache, prefix_len, q_ids, eos_ids, answer=ref_answer, cfg=cfg)
        k = kl(ref_lp, lp).cpu().numpy()
        R = {key: torch.stack([r[key] for r in recs]).cpu().numpy() for key in REC_KEYS}
        kept = R["kept"]
        k_loaded = np.ones_like(kept) if cfg.mode in ("fresh",) + run_reuse.reuse.VSKIP_MODES else kept
        spent[name] = bytes_of(cfg, kept, k_loaded, L, D, G)
        sparse = [l for l in range(L) if l not in cfg.dense_layers]
        out["conditions"][name] = {
            "answer": text, "retrieved": first_int(text) == pass_key, "answer_same": not forced, **spent[name],
            "kl_mean": float(k.mean()), "kl_x_noise": float(k.mean() / noise_kl.mean()),
            "kl_answer_x_noise": float(k[n_q - 1:].mean() / noise_kl[n_q - 1:].mean()),
            "kl_max": float(k.max()), "worst_step": int(np.argmax(k)),
            "covered_min_p01": float(np.quantile(R["covered_min"][:, sparse], 0.01)),
            "refresh_rate": float(R["refresh"][1:, sparse].mean()), "seconds": round(time.perf_counter() - t0, 1)}
        if head_rows is not None:
            S, _, H = kept.shape
            head_rows.append(pd.DataFrame({
                "depth": depth, "condition": name, "step": np.repeat(np.arange(S), L * H).astype(np.int16),
                "layer": np.tile(np.repeat(np.arange(L), H), S).astype(np.int16),
                "kv_head": np.tile(np.arange(H), S * L).astype(np.int16),
                **{key: R[key].reshape(-1).astype(np.float32)
                   for key in ("covered_min", "covered_stale_min", "refresh", "miss_p", "calib", "auto_ix",
                               "idx_cov", "m_after", "m_same")},
                "kl": np.repeat(k, L * H).astype(np.float32)}))
        c = out["conditions"][name]
        print(f"  {name:44s} {'ok  ' if c['retrieved'] else 'MISS'} {text!r:22s} KL x{c['kl_x_noise']:.2f} "
              f"bytes={c['bytes_total']:.3f} cov p01={c['covered_min_p01']:.3f} {c['seconds']}s", flush=True)
    del cache
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b", choices=list(common.MODELS))
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--iterations", type=int, default=10, help="depths 100 // n * (i + 1) %%, as in Quest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", nargs="*", default=[], help="conditions, see run_reuse.parse_cond")
    ap.add_argument("--depths", nargs="*", type=int, default=[], help="only these depths (percent)")
    ap.add_argument("--tag", default="", help="output name suffix")
    ap.add_argument("--save-heads", action="store_true", help="per-head records -> .heads.parquet")
    args = ap.parse_args()
    tok, model = common.load(args.model)
    STATE.vocab = vocab_classes(tok, model.device, model.config.vocab_size)
    n_garbage = garbage_for(tok, args.ctx - N_ANSWER)
    print(f"[passkey] {args.model} ctx {args.ctx}: filler of {n_garbage} characters", flush=True)
    runs, heads = [], ([] if args.save_heads else None)
    stem = OUT / (f"{args.model}_{args.ctx}" + (f"_{args.tag}" if args.tag else ""))
    depths = args.depths or [100 // args.iterations * (i + 1) for i in range(args.iterations)]
    for depth in depths:
        configs = [parse_cond(c) for c in args.cond]
        runs.append(run(tok, model, configs, args.ctx, depth, args.seed, n_garbage, heads))
        OUT.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(
            {"model": args.model, "ctx": args.ctx, "n_garbage": n_garbage, "runs": runs}, indent=1))
        if heads:
            pd.concat(heads).to_parquet(f"{stem}.heads.parquet")
    names = list(runs[0]["conditions"])
    print(f"\nreference retrieved {sum(r['ref_retrieved'] for r in runs)}/{len(runs)}")
    for n in names:
        cs = [r["conditions"][n] for r in runs]
        print(f"{n:44s} retrieved {sum(c['retrieved'] for c in cs)}/{len(cs)}  "
              f"KL x{np.exp(np.mean(np.log([c['kl_x_noise'] for c in cs]))):.2f} (gmean)  "
              f"bytes {np.mean([c['bytes_total'] for c in cs]):.3f}")


if __name__ == "__main__":
    main()
