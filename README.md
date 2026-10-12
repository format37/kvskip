# kvskip

Measurements toward a decode-time attention kernel that skips KV blocks nobody needs.

Stage 1 asks a plain question before any CUDA is written: during **decode** on real long
context, how much of the KV cache does each head actually use, and what does dropping the
rest cost? Models: Llama-3.2-1B / Llama-3.1-8B (bf16 safetensors), contexts 4k–32k,
three genres (prose, code, multi-turn dialog). Hardware: one RTX 4090.

Digest (Russian): [`docs/01-decode-sparsity-digest.md`](docs/01-decode-sparsity-digest.md) ·
plan: [`docs/01-decode-sparsity.md`](docs/01-decode-sparsity.md)

## Stage 1 in five lines (Llama-3.1-8B, 32k, decode)

- Half of the heads put 90% of their attention in 1–3 blocks of 64 tokens; reaching 99% takes ~31% of blocks per query head and ~50% per KV head (GQA union of 4 heads). The diffuse tail is functional: cutting it to 90% mass costs KL ×10.
- Oracle thresholding at block weight < 1e-4 skips 47% of KV blocks at the noise level of stock bf16 SDPA; < 1e-3 skips 84% at KL ×5; < 1e-2 skips 98% at ×66. Keeping each head's top-99% mass beats a fixed threshold (37% vs 24% skipped at 16k, same quality).
- A block's max QK score ranks blocks almost like the oracle (+5–6% blocks, Spearman 0.96), but the pre-exp cutoff `exp(M_b − m) < τ` needs ~2.5 nats of margin.
- Block selection is stable over time: the set chosen 16 steps earlier still covers 97% of the mass.
- Greedy text is a noisy metric: stock bf16 kernels diverge from the fp32 reference in 8 of 12 documents. Compare by KL against the bf16 noise floor.

Kernel direction: a two-pass split-K decode that reads all of K and skips V blocks per GQA union (−24% KV bytes free at 32k). Skipping K too needs a selection-reuse ablation first.

## Layout

```
src/
  probe_attn.py    "probe" attention for HF transformers 5.x: fp32 decode, stats, block masking
  metrics.py       per-head stats: entropy, sink mass, blocks-to-mass, GQA union, pre-exp, reuse
  common.py        model loading, prompt building, chunked prefill, decode step
  prepare_data.py  builds data/docs/{prose,code,dialog}.json from data/raw
  run_collect.py   prefill + 128 greedy decode steps -> per-step/layer/head parquet
  run_ablate.py    block masking (weight / pre-exp / top-p, per-head / union) -> KL vs fp32 reference
  analyze.py       parquet -> results/tables/*.{csv,md} and results/figures/*.png
  smoke_test.py    sdpa / eager / probe agreement, fp64 check of the decode path
data/docs/         exact inputs used (kept in git); data/raw is gitignored
results/           figures and tables are kept; raw parquet is regenerated
docs/              per-stage plan and digest
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
Data sources: Project Gutenberg #77931 (prose), `transformers/generation/utils.py` (code),
`HuggingFaceH4/ultrachat_200k` test split (dialog). `data/docs/` already contains the built inputs.

## Reproduce

```bash
.venv/bin/python src/smoke_test.py
.venv/bin/python src/run_collect.py --model 8b --ctx 4096 8192 16384 32768
.venv/bin/python src/run_ablate.py  --model 8b --ctx 16384 32768               # thresholds + bf16 noise floor
.venv/bin/python src/run_ablate.py  --model 8b --ctx 16384 32768 --grid mass   # top-p
.venv/bin/python src/analyze.py
```

8B at 32k peaks at ~20 GB; run one job at a time on a 24 GB card. Collection takes ~30 s per
document at 32k, the main ablation ~5 min per document.

## Status

- [x] Stage 1: decode attention statistics and masking ablation (1B, 8B, 4k–32k)
- [ ] Selection-reuse ablation: full pass every N ∈ {4, 8, 16, 32} steps
- [ ] Quest-style key bounds at block 16/32, locality of skipped blocks
- [ ] FA2 decode baseline; then the kernel (`src/kernels/`)
- [ ] Qwen3-8B, 64k+ contexts
