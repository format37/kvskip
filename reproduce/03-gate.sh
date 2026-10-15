#!/bin/bash
# Stage 3: the gate (decode step at bs 8-32 x 32k-64k in the layer-major lab), the planner in
# CUDA kernels, the token index for induction heads, Qwen3-8B.
# Digest: docs/03-gate-digest.md. One RTX 4090; one GPU job at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
DH=dh=2.5,5.2,16.5,27.1,26.3,25.1,31.2,6.7

$PY -I src/prepare_s3.py                                                 # 6 new documents (reads data/raw)
$PY src/make_forced.py                                                   # teacher tokens (fp32 probe at 32k): 6 min, GPU 20 GB
$PY bench/test_planner.py                                                # device planner vs Python planner: 2 min
$PY src/lab.py --ctx 32768 --tag s3                                      # phase 1 at 32k (Python planners): 6 min, GPU 21 GB, RAM 20 GB
$PY src/lab.py --ctx 65536 --tag s3                                      # 64k: 10 min, GPU 22 GB
for CTX in 32768 65536; do                                               # planner in kernels + index heads: 6 + 10 min
  $PY src/lab.py --ctx $CTX --tag s3h --conds sdpa dense refresh_fused refresh_fused_ix null \
      --index-heads 2.5 5.2 16.5 27.1 26.3 --index-union
done
$PY src/run_batch_e2e.py --ctx 32768 --batch 1 --kl                      # the lab vs the real model: 3 min, GPU 20 GB
$PY src/run_reuse.py --ctx 16384 32768 --grid s3a --cond estd:0.05:32:$DH estp:0.05:32:$DH       # 5 min
$PY src/run_reuse.py --ctx 32768 --grid s3a --doc prose2 code2 dialog2 --cond estd:0.05:32:$DH estp:0.05:32:$DH
$PY src/run_reuse.py --ctx 16384 32768 --grid s3b --cond estp:0.05:32 \
    estp:0.05:32:dh=25.1,31.2,6.7:ixu=2.5,5.2,16.5,27.1,26.3 estp:0.05:32:dh=25.1,31.2,6.7:ixub=2.5,5.2,16.5,27.1,26.3   # 8 min
$PY src/run_collect.py --model qwen3-8b --ctx 16384 32768                # 4 min, GPU 20 GB
$PY src/run_reuse.py --model qwen3-8b --ctx 16384 32768 --grid diag --cond timer:32 fresh estp:0.05:32   # 10 min
$PY src/calibrate_heads.py --model qwen3-8b --grid diag                  # CPU
$PY src/analyze_stage3.py                                                # CPU, 2 min
