# kvskip

A decode-time attention kernel that skips KV blocks nobody needs, and the measurements behind it.

Stage 1 asked a plain question before any CUDA was written: during **decode** on real long
context, how much of the KV cache does each head actually use, and what does dropping the
rest cost? Stage 2 decided which kernel to build and built its floor: a dense split-K
decode kernel at the bandwidth ceiling, block-skip modes, and the policy that chooses blocks.
Models: Llama-3.2-1B / Llama-3.1-8B (bf16 safetensors), contexts 4k–32k (kernels to 128k),
three genres (prose, code, multi-turn dialog). Hardware: one RTX 4090.

Digests (Russian): [stage 1](docs/01-decode-sparsity-digest.md) ([plan](docs/01-decode-sparsity.md)) ·
[stage 2](docs/02-core-digest.md) ([plan](docs/02-core.md))

## Stage 2 in five lines (Llama-3.1-8B, decode)

- Reusing the block selection with a periodic refresh fails: refreshing every 8 steps costs KL ×3.8–20 over the bf16 noise floor, because at a context switch ~100 of 248 KV heads lose coverage at once (layers 8–16 most).
- Refreshing a KV head when a cheap estimate of its lost mass exceeds 5% (exact block log-sum-exp from the last refresh, moved by the query drift projected on block key centroids; plus every 32 steps; plus 8 calibrated always-exact heads) keeps KL at ×1.3–1.5 of the noise floor and skips 39% of K+V bytes at 32k (dialog 53%); on three held-out documents ×1.4–1.5 and 38%. V-skip alone saves 25%.
- The dense kernel (raw CUDA, GQA-packed, cp.async, CUDA-core FMA) reaches 88–95% of the 4090's 1008 GB/s except at 8k/bs=1 (80%), against a streaming-read ceiling of 956 GB/s: 1.00–1.23× FA2 2.8.3 throughput, on par with cuDNN SDPA; error vs fp64 on real q/k/v 1.83e-3 (FA2 1.89e-3).
- Block-skip modes live in the same kernel family and their time follows the bytes read (50% of blocks: ×0.52, 10%: ×0.12). A single-pass V-skip that takes the normaliser from the previous step skips as much as the oracle (46% of V at 32k); a sink-anchored bound is safe but skips 35%.
- Chat template: dialog stays the sparsest genre on a second document (38–41% of blocks vs 53–59%); service tokens are not extra sinks (inserted into prose they change nothing), but removing the template from a dialog closes ~60% of the gap.

## Stage 1 in five lines (Llama-3.1-8B, 32k, decode)

- Half of the heads put 90% of their attention in 1–3 blocks of 64 tokens; reaching 99% takes ~31% of blocks per query head and ~50% per KV head (GQA union of 4 heads). The diffuse tail is functional: cutting it to 90% mass costs KL ×10.
- Oracle thresholding at block weight < 1e-4 skips 47% of KV blocks at the noise level of stock bf16 SDPA; < 1e-3 skips 84% at KL ×5; < 1e-2 skips 98% at ×66. Keeping each head's top-99% mass beats a fixed threshold (37% vs 24% skipped at 16k, same quality).
- A block's max QK score ranks blocks almost like the oracle (+5–6% blocks, Spearman 0.96), but the pre-exp cutoff `exp(M_b − m) < τ` needs ~2.5 nats of margin.
- Block selection is stable over time: the set chosen 16 steps earlier still covers 97% of the mass.
- Greedy text is a noisy metric: stock bf16 kernels diverge from the fp32 reference in 8 of 12 documents. Compare by KL against the bf16 noise floor.

Kernel direction after stage 1: a two-pass split-K decode that reads all of K and skips V blocks per GQA union (−24% KV bytes free at 32k); skipping K too needed a selection-reuse ablation first. Stage 2 ran it, see above.

## Layout

```
kernels/
  kvskip_kernel.cu   split-K decode (dense / masked block list / per-block stats / stats-only + P),
                     two-pass V-skip pass 2, single-pass V-skip, split combine, streaming-read probe
  kvskip_ext.cpp     torch bindings;  kvskip.h  launch parameters
  kvskip.py          Python front-end: build (cpp_extension, sm_89), decode(), plans, V-skip planners
bench/
  harness.py         L2 sweep + CUDA-event timing, fp64 references, bandwidth ceiling
  test_decode.py     correctness on real q/k/v dumps vs fp64, next to SDPA and FA2
  bench_decode.py    dense: SDPA (all backends) / FA2 / kvskip, bs {1,8,32} x ctx {8k..128k}
  bench_sparse.py    sparse modes on real masks + kept-fraction sweep
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
  analyze.py         stage 1 tables and figures;  analyze_stage2.py  stage 2
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
```

Llama weights are gated: accept the license on Hugging Face and `hf auth login` first.
Data sources: Project Gutenberg #77931 and #77919 (prose), `transformers/generation/utils.py` and
`trainer.py` (code), `HuggingFaceH4/ultrachat_200k` test split (dialog). `data/docs/` already
contains the built inputs.

FA2 (baseline only) has no wheel for torch 2.14 and its setup.py compiles with `-std=c++17`,
while torch 2.14 headers need C++20. Download the `flash-attn==2.8.3.post1` sdist, replace
`-std=c++17` by `-std=c++20` in its setup.py, then build the sm_80 kernels (they run on sm_89),
about 70 minutes on 8 cores:

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

8B at 32k peaks at ~20 GB; run one job at a time on a 24 GB card. Collection takes ~30 s per
document at 32k, the main ablation ~5 min per document, a reuse condition ~20 s, A3 ~2.5 min
per condition and document.

## Status

- [x] Stage 1: decode attention statistics and masking ablation (1B, 8B, 4k–32k)
- [x] Selection-reuse ablation (timer / triggers), mask locality, Quest at 16/32, perplexity, chat-template control
- [x] FA2 / SDPA baselines; dense kernel at the bandwidth ceiling; masked, stats and V-skip modes
- [ ] Fuse the planner (top-p selection, refresh trigger) into kernels; CUDA graph the decode step
- [ ] End-to-end throughput at bs 8–32 / 64k–128k; Qwen3-8B
