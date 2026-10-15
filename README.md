# kvskip

What does a long-context LLM actually read when it generates a token, and can a decode kernel
skip the rest? These are measurements of decode-time attention in Llama-3.1-8B, Qwen3-8B,
Qwen2.5-7B and Mistral-7B (plus Llama-3.2-1B) at 4k–64k context on prose, code and multi-turn
dialog, and the CUDA decode kernel built to measure and act on them. Everything runs on one
RTX 4090.

Stage plans and digests (Russian): [stage 1](docs/01-decode-sparsity-digest.md) ([plan](docs/01-decode-sparsity.md)) ·
[stage 2](docs/02-kernel-digest.md) ([plan](docs/02-kernel.md)) · [stage 3](docs/03-gate-digest.md) ([plan](docs/03-gate.md)) ·
[stage 4](docs/04-tails-digest.md) ([plan](docs/04-tails.md)) · [post draft](docs/post-draft.md)

## What the models do: three facts

**1. Attention has a compact body and a long tail, and the tail matters.** In half of the
heads, 90% of a decode step's attention lands in one to three blocks of 64 tokens. Reaching
99% of the mass takes about a third of the context per query head at 32k, and about half per
KV head, because the query heads sharing a KV head want different blocks. The tail is not
noise: dropping everything beyond 90% of the mass makes the output distribution ten times
further from exact than bf16 rounding does. The body is why attention looks sparse; the
tail is why skipping it is hard. The needed share falls as context grows (the 8B needs
~T^0.8 blocks).

**2. The model jumps instead of drifting.** On average the blocks a head needs barely change
from step to step: the set chosen 16 steps ago still covers 97% of today's mass. The average
hides events. At some steps the model switches to another part of the context, and up to
~100 of Llama's 248 sparse KV heads lose their blocks in the same step (Qwen3: 83 of 280,
Qwen2.5: 35 of 108). So a cache of "blocks this head needed recently" cannot be refreshed on
a timer: every 8 steps already costs 4–20× the bf16 noise. It works when each head refreshes
on an event, namely a cheap estimate of the attention mass it is missing. The estimate is
blind to rare single-step misses, and on Llama almost all of them come from one induction
head (layer 2, KV head 5), which attends to the token right after an earlier copy of the
current token.

**3. The body depends on the model family; the long tail is everywhere.** Llama-3.1-8B and
Mistral-7B park their spare attention on the first tokens (an attention sink, a median 0.66
and 0.39 of a head's mass) from layer 1 on. Qwen3-8B and Qwen2.5-7B have almost none: their
first layers (0–6 and 0–3) put nothing on the first tokens, the middle layers 0.02–0.1, and
they lean on the newest tokens instead. Qwen2.5 has no query/key normalisation, so the weak
sink belongs to the Qwen family, not to QK-norm. Every model has the long tail and the
jumps: 99% of a query head's mass takes 29–39% of the blocks at 32k (a KV head, the union
of its 4 or 7 query heads, 49–69%), and 19–40% of the KV heads can lose their blocks in one
step. The same trigger with the same calibration works on all four without tuning. A related
observation from stage 2: a chat is the sparsest genre (38–41% of blocks against 53–59% for
prose and code), and the chat template accounts for ~60% of that gap; the service tokens
themselves are not extra sinks.

## The mechanism

- **Stored set.** Each KV head keeps a set of 64-token blocks: the union over its query
  heads of the fewest blocks holding 99% of each one's mass, taken at the head's last
  refresh. A step reads K and V of the stored set, the first block and the two newest.
- **Refresh on an event.** Before the attention pass, a planner kernel estimates the mass
  outside that set: each unread block's exact log-sum-exp at the last refresh, moved by the
  query drift projected on the block's key centroid (centroids are 1/64 of K). A head whose
  estimate exceeds 5% (or whose set is 32 steps old) reads every block instead, and the block
  statistics of that same pass give its new set. Deciding before the pass means no double
  read and no host synchronisation.
- **Online calibration (stage 4).** A refreshing head reads every block, so it learns how
  much its stale set really missed, next to the estimate it decided with. It keeps a
  decaying maximum of the difference, c = max(0.95 c, real − estimated), and adds c to its
  estimate: a head that once missed badly stays sensitive for a long time. During the first
  32 steps every head refreshes at least every 8 steps, to collect samples. A head whose
  correction exceeds 0.1 joins the token index: blocks holding an earlier copy of the
  current token or the token right after one become part of its attended set, found by
  token id without reading K. This replaces the hand-picked list of always-exact heads
  (Llama) and the lower threshold (Qwen3) of stage 3. Layer 0 is always dense.
- **Kernels.** A raw-CUDA split-K decode kernel family (dense, masked block list, per-block
  statistics, V-skip) at 88–95% of the 4090's bandwidth, and the planner as two more
  kernels: `plan` (estimate, decision, block list) → masked attention with block statistics
  → `update` (new set, references, calibration). Three launches per layer, no host sync;
  identical decisions to the Python reference on every head-step.
- **Stateless alternative.** Single-pass V-skip reads all of K and skips V of blocks whose
  weight, against the previous step's normaliser, is below 1e-4. No planner, no
  calibration; it saves 15–29% of the decode step where the policy saves 18–34%.

## The numbers

Quality is the KL divergence of the next-token distribution from an exact fp32 reference, in
multiples of the same KL for stock bf16 SDPA (the noise floor), mean over 128 decode steps
and over documents (worst document in parentheses).

**Calibration instead of hand parameters** (exact model with simulated attention, teacher-forced;
3 documents per column: prose, code, dialog; held-out = 3 other documents):

| | Llama 16k | Llama 32k | Llama 32k, held-out | Qwen3 16k | Qwen3 32k |
|---|---|---|---|---|---|
| trigger alone | 109 (319) | 1.87 (2.6) | 4.69 (11) | 1.31 (1.5) | 1.52 (2.2) |
| stage 3: + 8 exact heads (Llama) / δ = 0.02 (Qwen3) | 1.27 (1.7) | 1.35 (1.5) | 1.30 (1.5) | 1.30 (1.9) | 1.24 (1.6) |
| **stage 4: online calibration** | **1.06 (1.5)** | **1.54 (1.8)** | **1.65 (2.4)** | **1.21 (1.3)** | **1.03 (1.1)** |
| K+V bytes read, stage 4 (stage 3) | 74% (74%) | 64% (64%) | 65% (64%) | 73% (76%) | 69% (72%) |

**Bytes, attention time and step time** (Llama-3.1-8B, the stage-4 policy in the CUDA
kernels, layer-major lab, 8 documents of 70k+ tokens, teacher-forced): the K+V blocks read
per step (layer 0 dense included), the attention part of the step (layer time minus the
same layer with attention returning zeros) and the whole decode step, dense kernel → policy.

| | K+V read | attention, ms | decode step, ms (HF eager) | KL × noise |
|---|---|---|---|---|
| bs=8, 32k | 62% | 32.0 → 21.6 (−33%) | 57.9 → 47.5 (−18%) | 1.25 |
| bs=32, 32k | 62% | 146.6 → 99.9 (−32%) | 172.5 → 125.9 (−27%) | 1.25 |
| bs=8, 64k | 57% | 69.7 → 44.4 (−36%) | 95.5 → 70.2 (−26%) | 1.38 |
| bs=32, 64k | 57% | 296.6 → 187.2 (−37%) | 322.8 → 213.3 (−34%) | 1.38 |

The stage-3 policy with its hand-picked heads gives the same step times within a point
(−19 / −27 / −28 / −35%) at KL ×1.41 / ×1.32; single-pass V-skip −15 / −20 / −25 / −29%.

Step times are HF eager: everything outside attention costs 26 ms per step against a 16 ms
ideal, so a tuned engine would give KV skipping a larger share of the step. At bs=1 there is
no gain: KV is 22% of the bytes, and three launches per layer cost more than they save.

**At equal bytes** (Llama-3.1-8B, 32k, 6 documents; bytes are K+V over all layers and steps,
including each method's metadata; KL as the geometric mean over documents, worst in parentheses):

| method | K+V bytes | KL × noise |
|---|---|---|
| refresh + online calibration (stage 4), top-p 0.99 | 64% | 1.53 (2.4) |
| refresh + 8 always-exact heads (stage 3) | 64% | 1.32 (1.5) |
| per-step oracle: best blocks by exact weight (needs all of K; not implementable) | 64% | 1.32 (1.8) |
| Quest, 16-token pages, bounds combined over the GQA group by softmax sum (its best variant here) | 64% | 4.7 (51) |
| Quest, 16-token min/max, 64-token loads | 64% | 12.3 (84) |
| static: the first step's best blocks, never updated (SnapKV / H2O-like) | 64% | 7.0 (28) |
| single-pass V-skip (K read in full) | 77% | 1.19 (1.8) |

The refresh policy is the best implementable method at its own operating point, and its
stage-3 form is as good as choosing the best blocks afresh on every step. Quest depends on the
document: its best variant beats the policy on one code file (×1.5 vs ×1.8) and loses ×17 on
prose and ×51 on another code file. Static selection breaks at jumps. Below the operating
point (lower top-p, threshold 1.04 − top-p) the policy degrades faster than per-step
selection: at 40% of the bytes ×18 against Quest ×35 and the oracle ×2.9; at 30% it ties
Quest (×61 vs ×59). Reuse is an operating-point method.

**Free running at 64k** (8 documents, batch 1, each mode decoding 128 steps on its own
tokens; KL against the exact model on the mode's own prefixes, in multiples of stock SDPA
running freely): the calibrated policy ×1.79 (worst document 3.6), the stage-3 policy ×1.81,
single-pass V-skip ×2.40. The per-step KL stays in one band over all 128 steps, so the error
does not accumulate through the generated tokens' KV. The greedy pick equals the exact
model's on the same prefix on 99.4% of the steps; where the text departs from the exact
continuation (4 of 8 documents, from step 17–55; stock SDPA: 2 of 8) it paraphrases ("has
crept" → "had crept") rather than drifts. The lab cannot do this (it runs a decode position
one layer at a time), so each document runs in the real model with the embedding table and
lm_head on the CPU to fit 8 GiB of KV next to the layers.

## Limitations

- One GPU, four ~8B models (the kernels and the gate on Llama-3.1-8B only), 6–8 documents
  per measurement, 128 decode steps each. The mean KL is tail-dominated: one step of one
  head can move it several-fold.
- Quality is mostly teacher-forced KL against an fp32 reference; perplexity over 2k tokens
  (stage 2) and free running at 64k (stage 4) are the checks beyond it.
- The step times at bs ≥ 8 come from the layer-major lab, which composes a step from real
  HF layer calls on real data; it matches the real model where both fit (±2%) and
  overestimates CPU-bound configurations.
- The planner kernels assume a GQA group of 4 and head dim 128 (Llama-3.1-8B, Qwen3-8B,
  Mistral-7B); Qwen2.5-7B (group of 7) runs only in the probe. Kernels are built for sm_89.
- No paged KV, continuous batching, CUDA graphs or prefill: this is a decode-attention study,
  not an engine.
- The calibration is learned per sequence: a head that misbehaves rarely can still miss once
  before it is caught (Llama code documents carry most of the remaining gap).

## Setup

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

Model weights go to `models/` (Llama and Mistral weights are gated: accept the license on
Hugging Face and log in first). Download them with `huggingface_hub.snapshot_download`, e.g.
`snapshot_download("meta-llama/Llama-3.1-8B-Instruct", local_dir="models/llama-3.1-8b")`; the
other directories are `models/llama-3.2-1b`, `models/qwen3-8b` (Qwen/Qwen3-8B),
`models/qwen2.5-7b` (Qwen/Qwen2.5-7B-Instruct) and `models/mistral-7b`
(mistralai/Mistral-7B-Instruct-v0.3, the `model-*.safetensors` shards only).

FA2 (the baseline, and the lab's prefill) has no wheel for torch 2.14, and its setup.py
compiles with `-std=c++17` while torch 2.14 headers need C++20. Download the
`flash-attn==2.8.3.post1` sdist, replace `-std=c++17` by `-std=c++20` in its setup.py, then
build the sm_80 kernels (they run on sm_89 as native SASS, no PTX JIT), about 70 minutes on
8 cores:

```bash
FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=5 NVCC_THREADS=1 \
    uv pip install --python .venv/bin/python --no-build-isolation ./flash_attn-2.8.3.post1
```

The kvskip extension builds itself on first use (`kernels/build/`, ~1 min, nvcc 12.5 + ninja).

Data: `data/docs/` holds the exact inputs (Project Gutenberg #77931, #77919, #77959; files of
transformers 5.19 and torch 2.14; UltraChat 200k test conversations). `src/prepare_*.py`
rebuild them from the raw downloads in `data/raw/` (not included).

## Reproduce

`./reproduce.sh N` runs stage N; `reproduce/NN-*.sh` lists each stage's commands with wall
time and peak GPU memory on one RTX 4090. Run one GPU job at a time: the 8B at 32k peaks at
~21 GB, the lab and free running at 64k at ~22.5 GB. Stage 4 takes about 5 hours.

## Layout

```
kernels/
  kvskip_kernel.cu   split-K decode (dense / masked block list / per-block stats / stats-only + P),
                     two-pass V-skip pass 2, single-pass V-skip, split combine, streaming-read probe
  kvskip_planner.cu  the refresh planner on the device: plan -> masked + stats -> update, online calibration
  kvskip_ext.cpp     torch bindings;  kvskip.h  launch parameters;  kvskip.py  Python front-end
bench/
  harness.py         L2 sweep + CUDA-event timing, fp64 references, bandwidth ceiling
  test_decode.py     correctness on real q/k/v dumps vs fp64, next to SDPA and FA2
  bench_decode.py    dense: SDPA (all backends) / FA2 / kvskip, bs {1,8,32} x ctx {8k..128k}
  bench_sparse.py    sparse modes on real masks + kept-fraction sweep
  test_planner.py    device planner vs the Python planner on real q/k/v (decisions, outputs, calibration)
src/
  probe_attn.py      "probe" attention for HF transformers 5.x: fp32 decode, stats, block masking
  metrics.py         per-head stats: entropy, sink mass, blocks-to-mass, GQA union, pre-exp, reuse
  reuse.py           block-set policies inside the probe: reuse / triggers / calibration / V-skip /
                     Quest / static / oracle top-k
  common.py          model loading, prompt building, chunked prefill, decode step
  prepare_data.py    data/docs/{prose,code,dialog}.json;  prepare_a4.py  held-out + template controls;
  prepare_s3.py      8 documents of 70k+ tokens;  make_forced.py  their teacher tokens
  run_collect.py     prefill + 128 greedy decode steps -> per-step/layer/head statistics (parquet)
  run_ablate.py      block masking (weight / pre-exp / top-p) -> KL vs the fp32 reference
  run_reuse.py       block-set policies -> KL, bytes (with metadata), per-head records
  run_ppl.py         perplexity of a 2k-token held-out continuation
  run_a4.py          sparsity / sink / service-token mass per document (chat-template control)
  dump_kv.py         real decode-time q/k/v for kernel tests
  kvskip_attn.py     HF attention "kvskip" on the CUDA kernels + planners + GrowingCache
  run_e2e.py         the kernels inside the model: KL vs reference, decode step time
  lab.py             layer-major lab: decode step time at bs x ctx beyond the card, KL at 64k
  run_batch_e2e.py   the real model, batched, where the KV fits (checks the lab)
  run_free.py        free-running generation at 64k (stage 4)
  calibrate_heads.py the always-exact heads of a model from one diagnostic run (stage 3)
  analyze*.py        tables and figures per stage;  smoke_test.py  sdpa / eager / probe agreement
reproduce/           one script per stage;  reproduce.sh  runs them
data/docs/           exact inputs (kept in git); data/raw and data/kv_dumps are gitignored
results/             figures and tables are kept; raw parquet/json is regenerated
docs/                per-stage plan (kickoff) and digest, post draft
```

## Stage summaries

### Stage 4 in five lines (Llama-3.1-8B, Qwen3-8B, Qwen2.5-7B, Mistral-7B; decode)

- One mechanism replaces the hand parameters: each head corrects its miss estimate by a decaying maximum of what its refreshes showed it really missed, after a short warm-up, and heads with a large correction join the token index. At the same bytes: Llama ×1.06 at 16k (8-head list ×1.27, nothing ×109) and ×1.54–1.65 at 32k (list ×1.30–1.35); Qwen3 ×1.21 / ×1.03 against ×1.30 / ×1.24 with its tuned threshold. The kickoff's ratio correction fails: the misses that matter are rare single steps (40–94% of a head's mass), not a biased estimate.
- The heads the correction singles out are those the offline diagnostics found: Llama's induction head (2, 5); Qwen3's induction and duplicate-token heads (5 of its 8 offline-calibrated heads). Layer 0 stays dense (Llama is worse without it; Qwen3 does not care).
- At equal bytes (Llama, 32k, 6 documents, metadata counted) the refresh policy is the best implementable method at its operating point: ×1.53 at 64% of the bytes against Quest ×4.7–22 and static selection ×7.0; choosing the best blocks afresh every step (an oracle) gives ×1.32. Pushed to lower budgets it degrades faster than per-step selection and ties Quest at 30%.
- Qwen2.5-7B has no QK-norm but as weak a sink as Qwen3, so QK-norm is not the cause; Mistral-7B sinks like Llama. The share of blocks for 99% of the mass is not one constant (29–39% per query head, 49–69% per KV head at 32k), but all four have the long tail and the jumps, and the calibrated trigger works on all four without tuning (×1.03–1.54).
- Free running at 64k does not drift (×1.79 the noise of stock SDPA running freely, flat over 128 steps; texts paraphrase where they part). In the CUDA kernels the calibrated policy keeps stage 3's step-time gains (−18 / −27% at 32k, −26 / −34% at 64k, bs 8 / 32) without a head list.

### Stage 3 in five lines (Llama-3.1-8B, decode, 8 documents of 70k+ tokens)

- bs=8/32k needs 32 GiB of KV next to 15 GB of weights, so the kickoff's grid cannot run on a 24 GB card. A layer-major lab runs the exact model one layer at a time over a batch, timing real HF layer calls on real data. It matches the real model where both fit (dense, bs=1/32k: 24.2 vs 24.7 ms).
- Decode step vs the dense kernel, with the refresh planner in CUDA kernels: −19% (bs=8, 32k), −27% (bs=32, 32k), −28% (bs=8, 64k), −35% (bs=32, 64k), against byte bounds of 26/34/36/42%. KL stays at 1.3–1.4× the bf16 noise. The planner kernels make the same decisions as the Python planner on every head-step, with bit-identical outputs.
- Deciding the refresh before the attention pass costs nothing in quality (KL ×1.33 vs ×1.35 at 32k, AUROC 0.917 vs 0.926) and removes both the double read and the host sync.
- Five of the eight always-exact heads are induction heads: 8–18% of their attention goes to the token right after an earlier copy of the current token (1.6% for an average head). A token index alone covers 69–92% of their mass, so it helps as an addition to the stored set rather than as a replacement.
- Qwen3-8B: the jumps and the trigger transfer, the head list does not; it has almost no attention sink and its dense layers are 1–3.

### Stage 2 in five lines (Llama-3.1-8B, decode)

- Reusing the block selection with a periodic refresh fails: refreshing every 8 steps costs KL ×3.8–20 over the bf16 noise floor, because at a context switch ~100 of 248 KV heads lose coverage at once (layers 8–16 most).
- Refreshing a KV head when a cheap estimate of its lost mass exceeds 5% keeps KL at ×1.3–1.5 of the noise floor and skips 39% of K+V bytes at 32k (dialog 53%). The estimate is the exact block log-sum-exp from the last refresh, moved by the query drift projected on block key centroids; heads also refresh every 32 steps, and 8 calibrated heads stay exact. On three held-out documents: ×1.4–1.5 and 38%. V-skip alone saves 25%.
- The dense kernel reaches 88–95% of the 4090's 1008 GB/s except at 8k/bs=1 (80%), against a streaming-read ceiling of 956 GB/s: 1.00–1.23× FA2 2.8.3 throughput, on par with cuDNN SDPA.
- Block-skip modes live in the same kernel family and their time follows the bytes read (50% of blocks: ×0.52, 10%: ×0.12). A single-pass V-skip that takes the normaliser from the previous step skips as much as the oracle (46% of V at 32k); a sink-anchored bound is safe but skips 35%.
- Chat template: dialog stays the sparsest genre on a second document (38–41% of blocks vs 53–59%); service tokens are not extra sinks (inserted into prose they change nothing), but removing the template from a dialog closes ~60% of the gap.

### Stage 1 in five lines (Llama-3.1-8B, 32k, decode)

- Half of the heads put 90% of their attention in 1–3 blocks of 64 tokens; reaching 99% takes ~31% of blocks per query head and ~50% per KV head (GQA union of 4 heads). The diffuse tail is functional: cutting it to 90% mass costs KL ×10.
- Oracle thresholding at block weight < 1e-4 skips 47% of KV blocks at the noise level of stock bf16 SDPA; < 1e-3 skips 84% at KL ×5; < 1e-2 skips 98% at ×66. Keeping each head's top-99% mass beats a fixed threshold (37% vs 24% skipped at 16k, same quality).
- A block's max QK score ranks blocks almost like the oracle (+5–6% blocks, Spearman 0.96), but the pre-exp cutoff `exp(M_b − m) < τ` needs ~2.5 nats of margin.
- Block selection is stable over time: the set chosen 16 steps earlier still covers 97% of the mass.
- Greedy text is a noisy metric: stock bf16 kernels diverge from the fp32 reference in 8 of 12 documents. Compare by KL against the bf16 noise floor.

## Status

- [x] Stage 1: decode attention statistics and masking ablation (1B, 8B, 4k–32k)
- [x] Stage 2: kernel (dense / masked / stats / V-skip), selection reuse, refresh trigger, perplexity, chat-template control
- [x] Stage 3: the gate at bs 8–32 / 32k–64k (layer-major lab), planner in CUDA kernels, token index, Qwen3-8B
- [x] Stage 4: online calibration instead of hand parameters, Quest / static / V-skip at equal bytes, Qwen2.5-7B, free running at 64k, repository for outside readers
- [ ] CUDA graph for the decode step (bs = 1); split count from the actual block lists; block 32
- [ ] Paged-KV integration in a serving engine; prefill

## License

MIT (see `LICENSE`). Model weights and the documents in `data/docs` keep their own licenses.
