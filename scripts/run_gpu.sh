#!/bin/bash
# Trains one run on a GPU node.
#
# Usage:
#   sbatch --export="ALL,CONFIG=configs/<name>.yaml,OVERRIDES=seed=1 weight_decay=0.6" \
#       scripts/run_gpu.sh
#
# OVERRIDES is optional (space-separated key=value pairs, passed to --set).

#SBATCH -J eigengrokking_gpu
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source scripts/slurm_env.sh

echo "Host: $(hostname)"
echo "Assigned GPU: ${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

: "${CONFIG:?CONFIG must be set (exported via sbatch --export)}"
# OVERRIDES is deliberately word-split into separate key=value arguments.
# shellcheck disable=SC2086
python -m eigengrok.train --config "${CONFIG}" --set ${OVERRIDES:-}
