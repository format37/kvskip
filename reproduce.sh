#!/bin/bash
# Reproduce one or more stages: ./reproduce.sh 1 [2 3 4]   (no argument: list the stages).
# Each stage's commands, with wall time and peak GPU memory on one RTX 4090, are in reproduce/NN-*.sh.
# Prerequisites: README "Setup" (venv, model weights in models/, FA2 for stages 2-4) and the
# downloaded sources in data/raw for the prepare_* steps (data/docs already holds their output).
set -euo pipefail
cd "$(dirname "$0")"
if [ $# -eq 0 ]; then
  for f in reproduce/*.sh; do echo "$(basename "$f" .sh | cut -c1-2 | sed 's/^0//')  $f  $(sed -n 2p "$f" | sed 's/^# //')"; done
  exit 0
fi
for n in "$@"; do
  f=$(ls reproduce/"$(printf %02d "$n")"-*.sh 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "no stage $n"; exit 1; }
  echo "== $f"
  bash "$f"
done
