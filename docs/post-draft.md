# Post draft: how much of attention can a decoder skip?

*Skeleton and figure list (stage 4). The repository is the technical appendix: every number
below comes from a table in `results/tables/` and the stage digest that explains it.*

## 1. The question, and what we actually learned

- At 32k context, one decode step of Llama-3.1-8B reads 4 GB of KV per sequence; at batch 8
  and 64k, 69 GB against 15 GB of weights. Attention is "sparse" in the folklore sense: most
  heads put most of their mass on a few tokens. How much of the KV can a kernel skip without
  changing the output, and what does it take to know which part?
- The short answer: about half of the blocks per KV head are needed for 99% of the mass, and
  the other half can be skipped; the hard part is that the needed half moves in jumps, so
  the kernel must notice when it has stopped being right.
- Hook figure: per-head blocks needed for 0.9 vs 0.99 of the mass (bimodal body, long tail).

## 2. Three facts about the models

1. **A compact body and a functional tail.** Half the heads put 90% of a step's mass in 1-3
   blocks of 64 tokens; reaching 99% takes ~50% of the blocks per KV head at 32k. Cutting the
   tail at 90% makes the output ten times further from exact than bf16 rounding does.
2. **Jumps, not drift.** On average the set chosen 16 steps ago still covers 97% of the
   mass; at some steps ~100 of 248 KV heads lose their blocks at once. A timer cannot keep
   up; an event trigger can.
3. **The body depends on the model family; the long tail is everywhere.** Llama-3.1-8B and
   Mistral-7B park their spare mass on the first tokens (an attention sink, median 0.66 and
   0.39 of a head's mass); Qwen3-8B and Qwen2.5-7B have almost none (0.13, 0.19) and lean on
   the newest tokens. Qwen2.5 has no QK-norm, so the weak sink is a family trait (stage 3 had
   guessed QK-norm; stage 4 refuted it). The share of blocks for 99% of the mass is not one
   constant (29-39% per query head, 49-69% per KV head at 32k), but all four have the long
   tail and the jumps, and one calibrated trigger works on all four without tuning.

Figures: `8b_32768_layer_head_heatmaps.png` (where the mass sits, layer x head),
`blocks_needed_hist.png` (body vs tail), `stage2_a1_jumps.png` (coverage collapses under a
timer), `stage4_models.png` (four models: blocks for 0.99 and sink by depth).

## 3. Why a timer fails and what works instead

- Reuse a block set and refresh it every N steps: N = 8 already costs 4-20x the bf16 noise.
  The damage concentrates at jump steps.
- Refresh a head when a cheap estimate of the mass it is missing exceeds 5%: the estimate
  moves the last refresh's exact block log-sum-exps by the query drift projected on block key
  centroids (1/64 of K). It is decided before the attention pass, so no host sync.
- What the estimate cannot see: rare, single-step misses, almost all in one induction head of
  Llama (layer 2, KV head 5): it attends to the token after an earlier copy of the current
  token, a target that moves every step and that a block centroid averages away.
- Stage 3 handled it with a hand-made list of 8 always-exact heads. Stage 4 replaces the list
  by an online correction: a refreshing head learns how much it really missed, and a
  decaying maximum of (real - estimated) keeps a head that once missed badly sensitive; a
  short warm-up collects the first samples, and heads with a large correction join the token
  index. Same bytes as the list: Llama x1.06 at 16k (list x1.27, nothing x109), x1.54-1.65 at
  32k (list x1.30-1.35); Qwen3 x1.21 / x1.03 against x1.30 / x1.24 for its tuned threshold.
- The obvious corrections fail, and why is the interesting part: a ratio or a mean of
  (real - estimated) stays small, because the head's estimate is fine on most steps and
  catastrophically wrong (40-94% missed) on rare ones. What matters is the tail of the error,
  not its mean.
- The correction finds the same heads the offline diagnostics found: Llama's induction head,
  Qwen3's induction and duplicate-token heads (5 of its 8 offline-calibrated heads).

Figures: `stage4_calib.png` (the variants per model and document set), `stage3_index_map.png`
(induction heads).

## 4. The kernel as an instrument and as a result

- Dense split-K decode kernel at 88-95% of the 4090's bandwidth (1.00-1.23x FA2 2.8.3);
  sparse modes in the same family, time follows bytes (50% of blocks: x0.52).
- The policy's planner in two CUDA kernels, no host sync; identical decisions to the Python
  reference on every head-step.
- Honest frame: at bs = 1 there is no gain (KV is 22% of the step's bytes and HF eager is
  launch-bound). Where KV dominates (bs 8-32, 32k-64k), the calibrated policy reads 57-62%
  of the K+V, cuts the attention time by 32-37% and the whole decode step by 18 / 27%
  (32k, bs 8 / 32) and 26 / 34% (64k), at KL 1.25-1.38x the bf16 noise - the same as the
  stage-3 policy with its hand-picked heads. Step times are HF eager: 26 ms of every step is
  outside attention against a 16 ms ideal.
- Free running at 64k (8 documents, each mode on its own tokens, KL to the exact model on the
  mode's own prefixes): the calibrated policy x1.79 the noise of stock SDPA running freely,
  flat over 128 steps (no accumulation); where its text departs from the exact one (4 of 8
  documents) it paraphrases ("has crept" -> "had crept"). Figure `stage4_free.png`.

Figures: `stage2_bench_dense.png`, `stage3_gate.png` (or the stage-4 gate table).

## 5. Comparison with Quest (and the rest) at equal bytes

- Same 6 documents, same bytes per step including each method's metadata (Llama, 32k),
  geometric mean of KL x noise: refresh with calibration x1.53 at 64% of the bytes; the
  per-step oracle (needs all of K) x1.32; Quest at its 16-token pages x4.7, at the kickoff's
  16-token bounds / 64-token loads x12.3; static top blocks x7.0; single-pass V-skip x1.19
  at 77%.
- Quest is document-dependent: it beats the policy on one code file (x1.5 vs x1.8) and loses
  x17-51 on prose and another code file. Static breaks at jumps, as fact 2 predicts.
- Reuse is an operating-point method: lowering the top-p to read less degrades it faster than
  per-step selection (at 40% of the bytes x18 vs Quest x35 and the oracle x2.9; at 30% it
  ties Quest).

Figure: `stage4_budget.png` (KL vs bytes).

## 6. What did not work, and what we do not know

- Single-pass V-skip is the stateless floor: -15...-29% at bs >= 8 with no planner at all.
- Ratio and mean-type online corrections of the estimate (stage 4); the token-pair index.
- Unknown: a real serving engine (paged KV, CUDA graphs, fused non-attention ops); bs = 1;
  prefill; models beyond ~8B; whether semantic segmentation of a prompt can induce sparsity
  the way the chat template does in dialogs.

## 7. The repository by stage

- Stage 1 (`docs/01-decode-sparsity*`): the statistics and the masking ablation.
- Stage 2 (`docs/02-kernel*`): the kernel, selection reuse, the refresh trigger.
- Stage 3 (`docs/03-gate*`): the gate at bs 8-32 / 32k-64k, the planner in kernels, Qwen3.
- Stage 4 (`docs/04-tails*`): calibration instead of hand parameters, Quest at equal
  bytes, Qwen2.5, free running; `reproduce/` has one script per stage.

## Figure list

| # | file | for section |
|---|---|---|
| 1 | `results/figures/blocks_needed_hist.png` | 1, 2 |
| 2 | `results/figures/8b_32768_layer_head_heatmaps.png` | 2 |
| 3 | `results/figures/stage2_a1_jumps.png` | 2, 3 |
| 4 | `results/figures/stage4_models.png` | 2 |
| 5 | `results/figures/stage3_index_map.png` | 3 |
| 6 | `results/figures/stage2_bench_dense.png` | 4 |
| 7 | `results/figures/stage3_gate.png` | 4 |
| 8 | `results/figures/stage4_budget.png` | 5 |
