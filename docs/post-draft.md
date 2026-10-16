# Post draft: how much of attention can a decoder skip?

*Skeleton and figure list (stages 4-5). The repository is the technical appendix: every number
below comes from a table in `results/tables/` and the stage digest that explains it.*

## 1. The question, and what we actually learned

- At 32k context, one decode step of Llama-3.1-8B reads 4 GB of KV per sequence; at batch 8
  and 64k, 69 GB against 15 GB of weights. Attention is "sparse" in the folklore sense: most
  heads put most of their mass on a few tokens. How much of the KV can a kernel skip without
  changing the output, and what does it take to know which part?
- The short answer: at 32k, 99% of the mass takes about a third of the context per query head
  (29-39% of the blocks on four models) and half to two thirds per KV head (49-69%: the query
  heads sharing a KV head want different blocks); the rest can be skipped. The hard part is that
  the needed set moves in jumps, so the kernel must notice when it has stopped being right.
- Hook figure: per-head blocks needed for 0.9 vs 0.99 of the mass (bimodal body, long tail).
- How to read the numbers (defined once, used everywhere below): quality is the KL divergence of
  the next-token distribution from an exact fp32 reference, divided by the same KL of stock bf16
  SDPA, "x noise". Two bf16 kernels differ from each other by x0.84-1.43 on the same documents
  (`results/tables/stage2_e2e.md`: kvskip's dense kernel vs SDPA), so anything below about x1.5 is
  within noise: x1.5 does not mean "50% worse", it means indistinguishable from a different
  bf16 kernel. The mean over 128 steps is tail-dominated: one bad step of one head can move it
  several-fold, so worst documents are given next to means.

## 2. Three facts about the models

1. **A compact body and a functional tail.** Half the heads put 90% of a step's mass in 1-3
   blocks of 64 tokens; reaching 99% takes 29% of the blocks per query head and 49% per KV head
   at 32k (Llama). Cutting the tail at 90% makes the output ten times further from exact than
   bf16 rounding does.
2. **Jumps, not drift.** On average the set chosen 16 steps ago still covers 97% of the
   mass; at some steps ~100 of 248 KV heads lose their blocks at once. A timer cannot keep
   up on three of the four models (a refresh every 32 steps: x10-15 the noise at 32k); Mistral
   is milder at 32k (x2.0) but not at 16k (x400 on prose). An event trigger keeps up on all
   four.
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
- Where it stops: the rest of Llama's gap at 32k is a head that misses about once per
  document ((5, 2) on code: one step, x92). Can the list be learned from traffic instead,
  by carrying each head's correction from one request to the next (9 documents, two
  orders)? Not reliably. The correction is a decaying maximum, so it fades within a
  document. Carried over, it saved code in one order (x1.84 -> x1.15) and not in the other
  (x2.08), and the trace shows why it saved it: not by sensitivity (at step 62 the head
  still does not refresh), but by moving one earlier refresh from step 44 to step 38, whose
  stored set happened to hold step 62's target. The same shifts make new single-step misses
  on other documents. Over 16 (document, order) pairs: x1.34 against x1.45 without carry
  (+0.8 p.p. of bytes); the hand-made list gives x1.18 at no extra bytes. Holding each
  head's worst miss as a lasting floor also reaches x1.18, by making ~70 of 248 heads exact
  (+8 p.p.).
- Why: the list's signal is not in the policy's own samples. A head learns its real miss
  only when it refreshes, at moments the policy chooses; (5, 2) shows 2 invisible losses in
  132 refresh samples over 9 documents, as many as a dozen harmless heads, and four of the
  eight list heads show none. The list came from a diagnostic in which 32-step-old sets are
  checked on every step. Online calibration replaces the list where a head's misses are
  visible to its own samples (Llama's (2, 5), Qwen3's heads), and such a head's index latch
  can be carried from request to request: on Quest's passkey, where (2, 5) never reaches the
  threshold, a latch carried over removes the policy's only bad steps (x14.6 -> x1.5, same
  bytes). For heads that miss between samples, an offline list stays the honest answer.

Figures: `stage4_calib.png` (the variants per model and document set), `stage3_index_map.png`
(induction heads), `stage5_carry.png` (the carried state, document by document).

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

## 5. Quest: two methods for two operating points

- **It is Quest.** On real decode-time q/k (3 documents, 3 layers, 4 steps, 7 budgets) our
  selection picks exactly the pages Quest's own evaluation code picks: 0 of 2.3M page decisions
  differ (`results/bench/quest_reference.md`).
- **It works where it is meant to.** Quest's own task with Quest's own protocol (passkey at 32k,
  10 depths, Llama-3.1-8B-Instruct, which Quest's repository also runs): every pass key
  retrieved at 256-4096 tokens per query head, as published. So does every other method at those
  bytes (per-step oracle, static selection, our policy at its lowest levels): the passkey does not
  separate methods (`stage5_passkey`).
- **Token budgets are not bytes.** 2048 tokens per query head is 6% of a 32k context, but a GQA
  kernel reads the union of the four query heads' pages (11%), the page min/max (1/16 of K+V) and
  two dense layers: 22.5% of the dense K+V bytes, counted the way we count ours.
- **What a small budget costs in distribution terms.** At 2048-4096 tokens Quest's output is
  x90-210 the bf16 noise on the passkey and x240-460 on our documents (geometric means), and its
  greedy token matches the exact one on 91-92% of the teacher-forced steps. Below ~40% of the
  bytes nothing implementable stays near bf16: even choosing the best blocks afresh every step
  (an oracle that needs all of K) is x4-8 at 20-30%.
- **The published configuration is the weaker way to spend bytes on a GQA model.** Choosing
  pages per KV head (the variant stage 4 compared against) is about ten times better at equal
  bytes: x47 vs x458 at 22.5%, x4.7 vs ~x46 at 64%. Almost all of it is the per-query-head
  choice: each head attends only its own pages while the kernel reads the union of four anyway
  (x458 -> x56 at 22.5%); one dense layer instead of two gives the remaining x1.2
  (`stage5_quest_decomp`). The comparison below uses the stronger variant.
- **Two operating points.** At 64% of the bytes our policy is within noise (x1.53, oracle x1.32),
  Quest's GQA variant is x4.7 and the published configuration ~x46. At Quest's budgets our
  policy does not work: it cannot get down to Quest's bytes on the passkey (39% at its lowest
  level), and on documents at 20% it is x168 against x47 for Quest's GQA variant at 22.5%.
  Quest is built to keep the task at small budgets, and does; the refresh policy is built for
  the point where the output is indistinguishable from bf16, and reaches it at 64% of the bytes
  (single-pass V-skip at 77%). Static selection breaks at jumps (x7 at 64%), as fact 2 predicts.

Figure: `stage5_budget.png` (left: KL vs bytes on the full scale, 15-72%; right: the passkey).

## 6. What did not work, and what we do not know

- Single-pass V-skip is the stateless floor: -15...-29% at bs >= 8 with no planner at all.
- Ratio and mean-type online corrections of the estimate (stage 4); the token-pair index.
- Carrying the calibration from request to request (stage 5): it moves the rare offender's
  single-step misses around instead of removing them; reuse below ~40% of the bytes (stage 4-5).
- Unknown: how to find rare offenders online (audit refreshes at random steps would need
  hundreds to thousands of documents); a real serving engine (paged KV, CUDA graphs, fused
  non-attention ops); bs = 1; prefill; models beyond ~8B; whether semantic segmentation of a
  prompt can induce sparsity the way the chat template does in dialogs.

## 7. The repository by stage

- Stage 1 (`docs/01-decode-sparsity*`): the statistics and the masking ablation.
- Stage 2 (`docs/02-kernel*`): the kernel, selection reuse, the refresh trigger.
- Stage 3 (`docs/03-gate*`): the gate at bs 8-32 / 32k-64k, the planner in kernels, Qwen3.
- Stage 4 (`docs/04-tails*`): calibration instead of hand parameters, Quest at equal
  bytes, Qwen2.5, free running; `reproduce/` has one script per stage.
- Stage 5 (`docs/05-quest*`): Quest checked against its own code and in its own task, the
  full byte scale, the calibration carried across requests.

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
| 8 | `results/figures/stage5_budget.png` | 5 |
