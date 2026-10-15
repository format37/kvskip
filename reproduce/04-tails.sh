#!/bin/bash
# Stage 4: online calibration instead of hand parameters, Quest / static / V-skip at equal bytes,
# Qwen2.5-7B, free-running generation at 64k, the calibrated policy in the lab.
# Digest: docs/04-tails-digest.md. One RTX 4090; one GPU job at a time.
# Qwen2.5-7B-Instruct goes to models/qwen2.5-7b (huggingface_hub.snapshot_download, 15 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
DH=dh=2.5,5.2,16.5,27.1,26.3,25.1,31.2,6.7                       # stage-3 always-exact heads (Llama)
LOGIT="estp:0.05:32:cal=0.3 estp:0.05:32:cal=0.3:warm=32.8"        # kickoff: log-odds (ratio) correction, EMA
ADD="estp:0.05:32:cal=0.3:kind=add estp:0.05:32:cal=0.5:kind=add estp:0.05:32:cal=0.3:kind=add:warm=32.8 \
     estp:0.05:32:cal=0.5:kind=add:warm=32.8 estp:0.05:32:cal=0.3:kind=add:warm=32.8:dl=none \
     estp:0.05:32:cal=0.3:kind=add:warm=32.8:aix=0.05"            # additive correction, EMA
PEAK="estp:0.05:32:cal=0.05:kind=peak estp:0.05:32:cal=0.02:kind=peak estp:0.05:32:cal=0.05:kind=peak:warm=32.8 \
      estp:0.05:32:cal=0.05:kind=peak:warm=32.8:aix=0.1 estp:0.05:32:cal=0.05:kind=peak:warm=32.8:dl=none"
FINAL=estp:0.05:32:cal=0.05:kind=peak:warm=32.8:aix=0.1           # the stage-4 policy

$PY bench/test_planner.py                                         # device vs Python planner, all calibration kinds: 3 min

# phase 1: calibration variants vs the manual parameters (probe; GPU 21 GB; ~20 s per condition and document)
$PY src/run_reuse.py --model 8b --ctx 16384 32768 --grid s4a --cond estp:0.05:32:$DH estp:0.05:32 $LOGIT \
    estp:0.05:32:cal=0.1 estp:0.05:32:cal=0.1:warm=32.8 estp:0.05:32:cal=0.3:hold=32 estp:0.05:32:cal=0.3:free \
    estp:0.05:32:cal=0.3:warm=32.8:dl=none                          # 25 min
$PY src/run_reuse.py --model 8b --ctx 16384 32768 --grid s4b --cond $ADD                                  # 15 min
$PY src/run_reuse.py --model 8b --ctx 16384 32768 --grid s4p --cond $PEAK                                 # 13 min
$PY src/run_reuse.py --model 8b --ctx 32768 --doc prose2 code2 dialog2 --grid s4b \
    --cond estp:0.05:32:$DH estp:0.05:32 $LOGIT $ADD $PEAK                                                 # 19 min
$PY src/run_reuse.py --model 8b --ctx 32768 --doc prose3 code3 dialog3 --grid s4c --cond estp:0.05:32:$DH \
    estp:0.05:32 estp:0.05:32:cal=0.05:kind=peak:warm=32.8 $FINAL estp:0.05:32:cal=0.05:kind=peak:warm=64.4:aix=0.1   # 9 min
$PY src/run_reuse.py --model qwen3-8b --ctx 16384 32768 --grid s4b --cond estp:0.05:32 estp:0.02:32 \
    estp:0.05:32:cal=0.3:warm=32.8 estp:0.05:32:cal=0.3:kind=add estp:0.05:32:cal=0.3:kind=add:warm=32.8 $PEAK \
    estp:0.05:32:cal=0.05:kind=peak:warm=32.8:dl=1,2,3            # 32 min, GPU 20 GB

# phase 2: equal bytes -- the policy at top-p 0.99 ... 0.4 (threshold delta = 1.04 - top-p: refresh when
# the estimate exceeds what the set was built to miss by 4 points), each matched in bytes by Quest
# (16-token min/max, loaded by 64), static top blocks and the per-step oracle; at 0.99 also Quest-64,
# other GQA combinations of the bounds and Quest's own 16-token pages; single-pass V-skip; stage 3
TAIL=cal=0.05:kind=peak:warm=32.8:aix=0.1
Q=""
for P in 0.99 0.9 0.8 0.6 0.4; do
  D=$(python3 -c "print(f'{1.04 - $P:.2f}'.rstrip('0').rstrip('.'))")
  SPEC=estp:$D:32:$TAIL; N=estp_delta${D}_age32_cal0.05w32a8peak_aix0.1
  [ "$P" != 0.99 ] && { SPEC=$SPEC:p=$P; N=${N}_p$P; }
  Q="$Q $SPEC quest:match=$N static:match=$N topk:match=$N"
  [ "$P" = 0.99 ] && Q="$Q quest:match=$N:sub=64 quest:match=$N:union=raw quest:match=$N:union=soft \
      quest:match=$N:union=rr quest:match=$N:native=1 quest:match=$N:native=1:union=soft"
  [ "$P" = 0.8 ] && Q="$Q quest:match=$N:union=soft quest:match=$N:native=1:union=soft"
done
Q="$Q vprev:0.0001 estp:0.05:32:$DH"
$PY src/run_reuse.py --model 8b --ctx 32768 --grid s4q --cond $Q                                         # 33 min
$PY src/run_reuse.py --model 8b --ctx 32768 --doc prose2 code2 dialog2 --grid s4q --cond $Q               # 33 min

# phase 3: Qwen2.5-7B (GQA 7:1, no QK-norm): layer map, sink, jumps, the trigger with and without calibration
$PY src/run_collect.py --model qwen2.5-7b --ctx 16384 32768                                               # 4 min, GPU 18 GB
$PY src/run_reuse.py --model qwen2.5-7b --ctx 16384 32768 --grid diag --cond timer:32 fresh estp:0.05:32 \
    estp:0.02:32 $FINAL                                                                                   # 14 min

# phase 3.2 (optional): Mistral-7B-Instruct-v0.3 (models/mistral-7b, HF-format shards only, 14.5 GB)
$PY src/run_collect.py --model mistral-7b --ctx 16384 32768                                               # 4 min, GPU 18 GB
$PY src/run_reuse.py --model mistral-7b --ctx 16384 32768 --grid diag --cond timer:32 estp:0.05:32 $FINAL  # 10 min

# phase 4: free-running generation at 64k, batch 1 per document (8 stage-3 documents)
$PY src/run_free.py --ctx 65536 --conds sdpa vskip1 refresh_s3 refresh_s4 \
    --calib 0.05 --calib-kind peak --calib-warm 32 8 --calib-auto-ix 0.1 --dense-layers 0                 # 35 min, GPU 22.5 GB

# the calibrated policy at the gate points (layer-major lab, 8 documents, bs 1 / 8 / 32)
for CTX in 32768 65536; do                                                                                # 7 + 12 min, GPU 22 GB
  $PY src/lab.py --ctx $CTX --tag s4 --conds sdpa dense vskip1 refresh_fused refresh_cal null \
      --calib 0.05 --calib-kind peak --calib-warm 32 8 --calib-auto-ix 0.1 --dense-layers 0
done
$PY src/analyze_stage4.py                                                                                 # CPU, 2 min
