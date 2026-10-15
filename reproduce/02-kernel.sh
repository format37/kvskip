#!/bin/bash
# Stage 2: the decode kernel (dense / masked / stats / V-skip), selection reuse and refresh triggers.
# Digest: docs/02-kernel-digest.md. One RTX 4090; one GPU job at a time.
# FA2 2.8.3 (baseline only) must be built first: README, "Setup" (~70 min, CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
DH=dh=2.5,5.2,16.5,27.1,26.3,25.1,31.2,6.7       # the 8 always-exact KV heads (layer.head) picked in A1

$PY src/dump_kv.py                                                       # real q/k/v for the kernel tests: 3 min, GPU 20 GB
$PY bench/test_decode.py                                                 # correctness vs fp64: 2 min (+1 min first build)
$PY bench/bench_decode.py --backends sdpa sdpa_all fa2 kvskip --tag dense   # 15 min, GPU up to 22 GB
$PY bench/bench_sparse.py                                                # 5 min
$PY src/run_reuse.py --ctx 16384 32768                                   # A1 timers / oracle + A2: 20 min, GPU 20 GB
$PY src/run_reuse.py --ctx 16384 32768 --grid diag                       # trigger signals, V-skip rules: 15 min
$PY src/run_reuse.py --ctx 16384 32768 --grid trig2 --cond estd:0.05:32:$DH estd:0.02:32:$DH   # 6 min
$PY -I src/prepare_a4.py                                                 # held-out documents (reads data/raw)
$PY src/run_reuse.py --ctx 32768 --grid heldout --doc prose2 code2 dialog2 \
    --cond fresh timer:16 oracle:0.98:32 estd:0.05:32:$DH                # 7 min
$PY src/run_a4.py                                                        # chat-template control: 8 min
$PY src/run_ppl.py --ctx 32768 --cond fresh vprev:0.0001 estd:0.05:32:$DH   # A3 perplexity: 1 h, GPU 21 GB
$PY src/run_e2e.py --ctx 32768 --delta 0.05 --dense-heads 2.5 5.2 16.5 27.1 26.3 25.1 31.2 6.7 --timing   # 10 min
$PY src/analyze_stage2.py                                                # CPU, 2 min
