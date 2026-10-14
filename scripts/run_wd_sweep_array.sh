#!/bin/bash

#SBATCH -J eigengrokking_wd_sweep
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --array=0-9
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 0:59:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_wd%x_%A_%a.out
#SBATCH --error=logs/slurm_wd%x_%A_%a.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/slurm_env.sh

: "${WEIGHT_DECAY:?WEIGHT_DECAY must be set (exported via sbatch --export)}"
INIT_SCALE="${INIT_SCALE:-1.0}"

echo "Host: $(hostname)"
echo "weight_decay=${WEIGHT_DECAY} init_scale=${INIT_SCALE} seed=${SLURM_ARRAY_TASK_ID}"

python -m eigengrok.train --config configs/mod91_wd_sweep.yaml --set \
    weight_decay="${WEIGHT_DECAY}" \
    initialization_scale="${INIT_SCALE}" \
    seed="${SLURM_ARRAY_TASK_ID}"
