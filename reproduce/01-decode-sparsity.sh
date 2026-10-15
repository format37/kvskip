#!/bin/bash
# Stage 1: decode-attention statistics and the block-masking ablation (Llama-3.2-1B, Llama-3.1-8B).
# Digest: docs/01-decode-sparsity-digest.md. One RTX 4090 (24 GB); run one GPU job at a time.
# Times are wall clock on that card; "GPU" is the peak memory of the step.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"

$PY src/smoke_test.py                                                    # 1 min, GPU 3 GB
$PY src/run_collect.py --model 1b --ctx 4096 8192 16384 32768 --out results/collect_v2   # 5 min, GPU 6 GB
$PY src/run_collect.py --model 8b --ctx 4096 8192 16384 32768 --out results/collect_v2   # 6 min (~30 s per document at 32k), GPU 20 GB
$PY src/run_ablate.py --model 8b --ctx 16384 32768                      # thresholds + bf16 noise floor: 25 min, GPU 20 GB
$PY src/run_ablate.py --model 8b --ctx 16384 32768 --grid mass          # top-p: 15 min, GPU 20 GB
$PY src/analyze.py --collect results/collect_v2                         # tables + figures: 2 min, CPU
