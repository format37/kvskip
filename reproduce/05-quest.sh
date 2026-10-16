#!/bin/bash
# Stage 5: Quest in its own regime (the published selection, passkey, the full byte scale) and the
# calibration state carried across documents.
# Digest: docs/05-quest-digest.md. One RTX 4090; one GPU job at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
TAIL=cal=0.05:kind=peak:warm=32.8:aix=0.1
F=estp:0.05:32:$TAIL                                              # the stage-4 policy
R2=estp_delta0.84_age32_cal0.05w32a8peak_aix0.1_p0.2
R1=estp_delta0.94_age32_cal0.05w32a8peak_aix0.1_p0.1

# phase 1.1: our Quest selection against Quest's own evaluation code (needs data/kv_dumps, src/dump_kv.py)
$PY bench/test_quest.py                                           # 10 s, GPU 4 GB

# phase 1.1-1.2: passkey retrieval with Quest's protocol at 32k, 10 depths (Quest at 256 ... 4096 tokens per
# query head; the refresh policy at top-p 0.99 / 0.4 / 0.2; oracle, static and stage 4's Quest at Quest-2048's bytes)
$PY src/run_passkey.py --ctx 32768 --iterations 10 --cond questp:256 questp:512 questp:1024 questp:2048 questp:4096 \
    $F estp:0.64:32:$TAIL:p=0.4 estp:0.84:32:$TAIL:p=0.2 topk:match=questp2048 static:match=questp2048 \
    quest:match=questp2048:native=1:union=soft                    # 50 min, GPU 21 GB

# phase 1.3: the full byte scale on stage 4's 6 documents (Quest as published at 512 ... 16384 tokens, the oracle at
# its bytes, stage 4's Quest variant at two of them, the refresh policy at top-p 0.2 / 0.1 and the oracle there);
# the points above 26% of the bytes come from stage 4 (grid s4q, reproduce/04-tails.sh)
$PY src/run_reuse.py --model 8b --ctx 32768 --doc prose code dialog prose2 code2 dialog2 --grid s5q --cond \
    questp:512 questp:1024 questp:2048 questp:4096 questp:8192 questp:12288 questp:16384 \
    topk:match=questp1024 topk:match=questp2048 topk:match=questp4096 topk:match=questp8192 topk:match=questp16384 \
    quest:match=questp2048:native=1:union=soft quest:match=questp8192:native=1:union=soft \
    estp:0.84:32:$TAIL:p=0.2 topk:match=$R2 estp:0.94:32:$TAIL:p=0.1 topk:match=$R1              # 34 min, GPU 21 GB

# phase 2: the calibration state across documents, forward and reverse order
CARRY="$F $F:carry=0.5 $F:carry=0.8 $F:carry=1 $F:carry=1:cw=0 $F:carry=1:ckind=latch $F:carry=0.8:ckind=floor"
$PY src/run_carry.py --model 8b --ctx 32768 --docs prose code dialog prose2 code2 dialog2 prose3 code3 dialog3 \
    --cond $CARRY                                                 # 55 min, GPU 21 GB
$PY src/run_carry.py --model qwen3-8b --ctx 32768 --docs prose code dialog prose2 code2 dialog2 \
    --cond $CARRY                                                 # 44 min, GPU 22.5 GB

# diagnostics: which head makes the policy's single-step spikes on the passkey (depths 30 / 50 / 60, with the
# stage-3 list next to it); why Quest as published is weaker than stage 4's GQA variant at equal bytes (pages per
# KV head with layers 0-1 dense); head (5, 2) step by step through prose -> code under the carry variants
DH=dh=2.5,5.2,16.5,27.1,26.3,25.1,31.2,6.7
$PY src/run_passkey.py --ctx 32768 --depths 30 50 60 --tag diag --save-heads --cond $F estp:0.05:32:$DH   # 4 min
$PY src/run_reuse.py --model 8b --ctx 32768 --doc prose code dialog prose2 code2 dialog2 --grid s5d --cond \
    questp:2048 quest:match=questp2048:native=1:union=soft:dl=0,1 quest:match=questp2048:native=1:union=soft  # 8 min
$PY src/run_carry.py --model 8b --ctx 32768 --docs prose code --orders fwd --tag trace --trace 5.2 2.5 \
    --cond $F $F:carry=0.5 $F:carry=0.8 $F:carry=1 $F:carry=1:cw=0                                        # 5 min
$PY src/run_passkey.py --ctx 32768 --depths 30 50 60 --tag latch25 --cond $F:ixu=2.5   # (2, 5) on the index from step 0: 2 min

$PY src/analyze_stage5.py                                         # CPU, 1 min
