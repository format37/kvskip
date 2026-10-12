# kvskip

What does a long-context LLM actually read when it generates a token, and can a decode kernel
skip the rest? These are measurements of decode-time attention in Llama-3.1-8B and Llama-3.2-1B
(bf16) at 4k–32k context on prose, code and multi-turn dialog, plus the CUDA decode kernel
built to measure and act on them. All runs use one RTX 4090.

Digests (Russian): [stage 1](docs/01-decode-sparsity-digest.md) ([plan](docs/01-decode-sparsity.md)) ·
[stage 2](docs/02-core-digest.md) ([plan](docs/02-core.md)) · [stage 3](docs/03-win-digest.md) ([plan](docs/03-win.md))

## What the model does: three facts

**1. Attention has a compact body and a long tail, and the tail matters.** In half of the
heads, 90% of a decode step's attention lands in one to three blocks of 64 tokens. Those are
the first tokens of the context (the attention sink) and the newest ones. The last 10% is
spread thinly over everything else. Reaching 99% of the mass takes about a third of the
context per query head at 32k. A KV head needs about half, because the four query heads
sharing it want different blocks. The tail is not noise: dropping everything beyond 90% of
the mass makes the output distribution ten times further from exact than bf16 rounding does.
The body is why attention looks sparse; the tail is why skipping it is hard. The needed
share falls as context grows (the 8B needs ~T^0.8 blocks). Layer 0 reads almost everything,
and the heads that really read the content sit in a band of middle layers, roughly 8–16.
What the body is depends on the model. Qwen3-8B, which normalizes queries and keys, has
almost no sink: its layers 0–6 put nothing on the first tokens, and it leans on the newest
ones instead. Its layer 0 is not dense either; layers 1–3 are. Yet it needs the same share
of blocks for 99% of the mass (50% per KV head at 32k, against 49%).

**2. The model jumps instead of drifting.** On average the blocks a head needs barely change
from step to step: the set chosen 16 steps ago still covers 97% of today's mass. The average
hides events. At some steps the model switches to another part of the context, and up to
~100 of its 248 sparse KV heads lose their blocks in the same step, against 3–10 on a typical
step, most of them in layers 8–16. Qwen3-8B jumps the same way (up to 86 of 280 heads at
once). So a cache of "blocks this head needed recently" can't be refreshed on a timer:
refreshing every 8 steps already costs 4–20× the bf16 noise. It works when each head
refreshes on an event, namely a cheap estimate of the attention mass it is missing. The
estimate misses a few heads, and five of Llama's worst are induction heads: they attend to
the token right after an earlier copy of the current token, a target that moves with every
token and that a lookup by token id finds without reading keys. Qwen3 has no such dominant
head; there the estimate alone keeps quality.

**3. Sparsity follows how the context splits into parts, not the special tokens.** A chat
is the sparsest genre: two different dialogs need 38–41% of blocks for 99% of the mass,
against 53–59% for prose and code. The template's service tokens are not extra attention
sinks. Real turn boundaries inserted into continuous prose every two paragraphs leave its
sparsity unchanged (57% → 57%). Yet the same dialog without the template, as a plain
"User: … / Assistant: …" transcript, needs 50%, so the template accounts for ~60% of the gap.
The dialog here is 28 unrelated conversations, so its turns are also separate in content.
With the template, the model seems to read it turn by turn and look mostly into the current
turn. In prose the same markers give it nothing to separate. Whether semantic segments, such
as unrelated chapters, can make prose as sparse is open.

## The kernel, and where speed stands

The kernel is the tool that made these measurements cheap and turns them into bytes.

- **Dense:** a raw-CUDA split-K decode kernel (GQA-packed, cp.async, CUDA-core FMA). It
  reaches 88–95% of the 4090's 1008 GB/s, which is the card's streaming-read ceiling
  (956 GB/s). That is 1.00–1.23× FA2 2.8.3 throughput, on par with cuDNN SDPA. Error vs fp64
  on real q/k/v is 1.83e-3 (FA2: 1.89e-3). FA2 was built from source for sm_80 because
  torch 2.14 has no wheel. The binary holds only sm_80 SASS (72 cubins, no PTX), which runs
  natively on the 4090's sm_89, so the baseline involves no JIT.
- **Sparse modes** live in the same kernel family, and their time follows the bytes read
  (50% of blocks: ×0.52).
- **Policy:** skip K and V outside a stored block set, and refresh a KV head when its estimated
  missed mass exceeds 5%. This skips 39% of K+V bytes at 32k (38% on held-out documents) at
  1.3–1.5× the bf16 noise. Perplexity over 2k continuation tokens matches exact attention.
- **End-to-end speed: no gain at bs=1, a real one where KV dominates.** At bs=1 and 32k,
  attention is ~18% of a decode step (4.3 GB of KV against 15 GB of weights), so skipping
  40% of the KV can save at most ~7% per token. At bs=8 and 64k, KV is 82% of the bytes and
  attention 73% of the step. There the refresh policy, with its planner moved from Python into
  two CUDA kernels, cuts the whole decode step by 28% (95.8 → 69.1 ms; the byte bound is 36%).
  At bs=32 and 64k it saves 35%, and at bs=8 and 32k 19%. The planner left in Python made the
  same step 27–73% slower. Single-pass V-skip, which needs no planner, saves 15–29% at bs ≥ 8.

## Stage summaries

### Stage 3 in five lines (Llama-3.1-8B, decode, 8 documents of 70k+ tokens)

- bs=8/32k needs 32 GiB of KV next to 15 GB of weights, so the kickoff's grid cannot run on a 24 GB card. A layer-major lab runs the exact model one layer at a time over a batch, timing real HF layer calls on real data. It matches the real model where both fit (dense, bs=1/32k: 24.2 vs 24.7 ms).
- Decode step vs the dense kernel, with the refresh planner in CUDA kernels: −19% (bs=8, 32k), −27% (bs=32, 32k), −28% (bs=8, 64k), −35% (bs=32, 64k), against byte bounds of 26/34/36/42%. KL stays at 1.3–1.4× the bf16 noise. The planner kernels make the same decisions as the Python planner on every head-step, with bit-identical outputs.
- Deciding the refresh before the attention pass costs nothing in quality (KL ×1.33 vs ×1.35 at 32k, AUROC 0.917 vs 0.926) and removes both the double read and the host sync.
- Five of the eight always-exact heads are induction heads: 8–18% of their attention goes to the token right after an earlier copy of the current token (1.6% for an average head). A token index alone covers 69–92% of their mass, so it helps as an addition to the stored set rather than as a replacement.
- Outside attention, eager HF runs at ~60% of the bandwidth limit (26 ms per step against a 16 ms ideal); a tuned engine would give KV skipping a larger share of the step.

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

## Layout

```
kernels/
  kvskip_kernel.cu   split-K decode (dense / masked block list / per-block stats / stats-only + P),
                     two-pass V-skip pass 2, single-pass V-skip, split combine, streaming-read probe
  kvskip_ext.cpp     torch bindings;  kvskip.h  launch parameters
  kvskip.py          Python front-end: build (cpp_extension, sm_89), decode(), plans, V-skip planners
  kvskip_planner.cu  stage 3: the refresh planner on the device (plan -> masked + stats -> update)
bench/
  harness.py         L2 sweep + CUDA-event timing, fp64 references, bandwidth ceiling
  test_decode.py     correctness on real q/k/v dumps vs fp64, next to SDPA and FA2
  bench_decode.py    dense: SDPA (all backends) / FA2 / kvskip, bs {1,8,32} x ctx {8k..128k}
  bench_sparse.py    sparse modes on real masks + kept-fraction sweep
  test_planner.py    stage 3: device planner vs the Python planner on real q/k/v (same decisions, outputs)
src/
  probe_attn.py      "probe" attention for HF transformers 5.x: fp32 decode, stats, block masking
  metrics.py         per-head stats: entropy, sink mass, blocks-to-mass, GQA union, pre-exp, reuse
  reuse.py           stage 2: selection reuse / refresh triggers / V-skip rules / mask locality (in the probe)
  common.py          model loading, prompt building, chunked prefill, decode step
  prepare_data.py    builds data/docs/{prose,code,dialog}.json from data/raw
  prepare_a4.py      builds data/docs/a4/*.json (new documents, chat-template controls)
  run_collect.py     prefill + 128 greedy decode steps -> per-step/layer/head parquet
  run_ablate.py      block masking (weight / pre-exp / top-p, per-head / union) -> KL vs fp32 reference
  run_reuse.py       stage 2 A1/A2: reuse policies -> KL, bytes, per-head coverage records
  run_ppl.py         stage 2 A3: perplexity of a 2k-token held-out continuation
  run_a4.py          stage 2 A4: sparsity / sink / service-token mass per document
  dump_kv.py         stage 2 B0: real decode-time q/k/v for kernel tests
  kvskip_attn.py     HF attention "kvskip" on the CUDA kernels + refresh planner + GrowingCache
  run_e2e.py         the kernels inside the model: KL vs reference, decode step time
  prepare_s3.py      stage 3: 8 documents of 70k+ tokens (data/docs/s3);  make_forced.py  their decode tokens
  lab.py             stage 3: layer-major lab -- decode step time at bs x ctx beyond the card, KL at 64k
  run_batch_e2e.py   stage 3: the real model, batched, where the KV fits (checks the lab)
  calibrate_heads.py stage 3: the always-exact heads of a model from one diagnostic run
  analyze.py         stage 1 tables and figures;  analyze_stage2.py / analyze_stage3.py  stages 2 / 3
  smoke_test.py      sdpa / eager / probe agreement, fp64 check of the decode path
data/docs/           exact inputs used (kept in git); data/raw and data/kv_dumps are gitignored
results/             figures and tables are kept; raw parquet/json is regenerated
docs/                per-stage plan and digest
```

## Setup

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python "transformers>=5.19" safetensors huggingface_hub numpy \
    pandas pyarrow scipy matplotlib tabulate accelerate ninja
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir models/llama-3.2-1b
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b
hf download Qwen/Qwen3-8B --local-dir models/qwen3-8b          # stage 3
```

Llama weights are gated: accept the license on Hugging Face and `hf auth login` first.
Data sources: Project Gutenberg #77931 and #77919 (prose), `transformers/generation/utils.py` and
`trainer.py` (code), `HuggingFaceH4/ultrachat_200k` test split (dialog). `data/docs/` already
contains the built inputs.

FA2 (baseline only) has no wheel for torch 2.14 and its setup.py compiles with `-std=c++17`,
while torch 2.14 headers need C++20. Download the `flash-attn==2.8.3.post1` sdist, replace
`-std=c++17` by `-std=c++20` in its setup.py, then build the sm_80 kernels (they run on sm_89
as native SASS, no PTX JIT), about 70 minutes on 8 cores:

```bash
uv pip install --python .venv/bin/python einops packaging psutil wheel setuptools
FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=5 NVCC_THREADS=1 \
    uv pip install --python .venv/bin/python --no-build-isolation ./flash_attn-2.8.3.post1
```

The kvskip extension builds itself on first use (`kernels/build/`, ~1 min, nvcc 12.5 + ninja).

## Reproduce

Stage 1:

```bash
.venv/bin/python src/smoke_test.py
.venv/bin/python src/run_collect.py --model 8b --ctx 4096 8192 16384 32768
.venv/bin/python src/run_ablate.py  --model 8b --ctx 16384 32768               # thresholds + bf16 noise floor
.venv/bin/python src/run_ablate.py  --model 8b --ctx 16384 32768 --grid mass   # top-p
.venv/bin/python src/analyze.py
```

Stage 2 (full list with timings in the digest, §10):

```bash
.venv/bin/python src/dump_kv.py && .venv/bin/python bench/test_decode.py
.venv/bin/python bench/bench_decode.py --backends sdpa sdpa_all fa2 kvskip --tag dense
.venv/bin/python bench/bench_sparse.py
.venv/bin/python src/run_reuse.py --ctx 16384 32768                 # A1 timers/oracle + A2
.venv/bin/python src/run_reuse.py --ctx 16384 32768 --grid diag     # trigger signals, V-skip rules
DH=dh=2.5,5.2,16.5,27.1,26.3,25.1,31.2,6.7
.venv/bin/python src/run_reuse.py --ctx 16384 32768 --grid trig2 --cond estd:0.05:32:$DH
.venv/bin/python -I src/prepare_a4.py && .venv/bin/python src/run_a4.py
.venv/bin/python src/run_ppl.py --ctx 32768 --cond fresh vprev:0.0001 estd:0.05:32:$DH
.venv/bin/python src/run_e2e.py --ctx 32768 --delta 0.05 --dense-heads 2.5 5.2 16.5 27.1 26.3 25.1 31.2 6.7 --timing
.venv/bin/python src/analyze_stage2.py
```

Stage 3 (full list in the digest, §9):

```bash
.venv/bin/python -I src/prepare_s3.py && .venv/bin/python src/make_forced.py
.venv/bin/python bench/test_planner.py
.venv/bin/python src/lab.py --ctx 32768 --tag s3 && .venv/bin/python src/lab.py --ctx 65536 --tag s3
.venv/bin/python src/lab.py --ctx 65536 --tag s3h --conds sdpa dense refresh_fused refresh_fused_ix null \
    --index-heads 2.5 5.2 16.5 27.1 26.3 --index-union
.venv/bin/python src/run_batch_e2e.py --ctx 32768 --batch 1 --kl
DH=dh=2.5,5.2,16.5,27.1,26.3,25.1,31.2,6.7
.venv/bin/python src/run_reuse.py --ctx 16384 32768 --grid s3a --cond estd:0.05:32:$DH estp:0.05:32:$DH
.venv/bin/python src/run_collect.py --model qwen3-8b --ctx 16384 32768
.venv/bin/python src/run_reuse.py --model qwen3-8b --ctx 16384 32768 --grid diag --cond timer:32 fresh estp:0.05:32
.venv/bin/python src/calibrate_heads.py --model qwen3-8b --grid diag
.venv/bin/python src/analyze_stage3.py
```

8B at 32k peaks at ~20 GB; run one job at a time on a 24 GB card. Collection takes ~30 s per
document at 32k, the main ablation ~5 min per document, a reuse condition ~20 s, A3 ~2.5 min
per condition and document.

## Status

- [x] Stage 1: decode attention statistics and masking ablation (1B, 8B, 4k–32k)
- [x] Selection-reuse ablation (timer / triggers), mask locality, Quest at 16/32, perplexity, chat-template control
- [x] FA2 / SDPA baselines; dense kernel at the bandwidth ceiling; masked, stats and V-skip modes
- [x] Stage 3 gate: decode step at bs 8–32 / 32k–64k (layer-major lab), step decomposition, byte bound vs actual
- [x] Refresh trigger before the attention pass; token index for the induction heads
- [x] Planner in CUDA kernels (no host sync), checked against the Python planner
- [x] Qwen3-8B: layer map, jumps, trigger without calibration, automatic head calibration
- [ ] CUDA graph for the decode step; split count from the actual block lists; block 32
- [ ] Paged-KV integration in a serving engine; Qwen3 in the lab (q/k norm, YaRN for 64k)
