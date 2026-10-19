#!/bin/bash
# Stage 6 (paper): task accuracy on RULER at 32k, the top-p ablation table, the paper's figures.
# One RTX 4090; one GPU job at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
TAIL=cal=0.05:kind=peak:warm=32.8:aix=0.1
F=estp:0.05:32:$TAIL                                              # the stage-4 policy
FN=estp_delta0.05_age32_cal0.05w32a8peak_aix0.1                   # ... its name, for match=

# RULER data (github.com/NVIDIA/RULER at commit c3f5e3b) in its own venv: the generators need wonderwords,
# nltk, scipy and an older transformers; inputs are Paul Graham's essays (niah_multivalue) and SQuAD dev
# v2.0 (qa_1), fetched by RULER's own scripts. CPU only, ~5 min plus ~4 min of downloads.
R=data/raw/ruler-src
if [ ! -d $R ]; then
  git clone https://github.com/NVIDIA/RULER.git $R && git -C $R checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
fi
uv venv data/raw/ruler-venv --python 3.11
uv pip install --python data/raw/ruler-venv/bin/python "transformers==4.44.2" wonderwords nltk scipy pyyaml tqdm \
    tenacity numpy html2text beautifulsoup4
(cd $R/scripts/data/synthetic/json && wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad.json \
    && ../../../../../ruler-venv/bin/python download_paulgraham_essay.py)
for T in niah_multikey_2 niah_multivalue vt fwe qa_1; do
  (cd $R/scripts/data && PATH="$PWD/../../../ruler-venv/bin:$PATH" python prepare.py --save_dir "$OLDPWD/data/raw/ruler-data" \
      --benchmark synthetic --task $T --tokenizer_path "$OLDPWD/models/llama-3.1-8b" --tokenizer_type hf \
      --max_seq_length 32768 --model_template_type meta-llama3 --num_samples 100)
done

# RULER at 32k, Llama-3.1-8B-Instruct, samples 0-49 of each task: the question and the answer go through the sparse
# decode path. Both operating points: the policy at top-p 0.99 (with Quest per KV head and the oracle at its bytes,
# and with the induction head (2, 5) latched to the token index from the first step, as a latch carried over from
# earlier requests would be), and Quest as published at 2048 tokens per query head (with Quest per KV head and the
# oracle at its bytes, and the policy at its lowest level, top-p 0.2).
$PY src/run_ruler.py --n 50 --tag main --cond $F $F:ixu=2.5 quest:match=$FN:native=1:union=soft topk:match=$FN \
    questp:2048 quest:match=questp2048:native=1:union=soft topk:match=questp2048 estp:0.84:32:$TAIL:p=0.2   # 6 h 33 min

$PY src/analyze_paper.py                                          # CPU, 1 min: tables (paper_*) and figures
