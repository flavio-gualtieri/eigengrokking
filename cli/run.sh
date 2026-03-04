#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

cd ~/work/eigengrokking
source ~/work/.venv/bin/activate

mkdir -p logs

N=${1:-1000}   # or set this to len(EXPERIMENTS)
for i in $(seq 0 $((N-1))); do
  ts=$(date +"%Y%m%d_%H%M%S")
  echo "=== exp_index=$i @ $ts ==="
  python -u main.py --exp_index "$i" 2>&1 | tee "logs/exp_${i}_${ts}.log"
done
