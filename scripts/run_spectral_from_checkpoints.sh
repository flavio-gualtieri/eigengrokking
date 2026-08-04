#!/bin/bash
# Runs scripts/spectral_from_checkpoints.py as a single GPU SLURM job.
#
# Usage:
#   sbatch --export="ALL,RUN_DIR=<run base dir>,STEPS=<step spec>,OUT=<parquet path>" \
#       scripts/run_spectral_from_checkpoints.sh
#
# RUN_DIR must be a run's base directory (contains checkpoints/ and
# config_full.json -- see project_io/dir_making.py). STEPS is either
# "start:stop:stride" or a comma-separated list, per
# `python scripts/spectral_from_checkpoints.py --help`.

#SBATCH -J spectral_from_ckpt
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 0:59:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/grok-env
set -u

: "${RUN_DIR:?RUN_DIR must be set (exported via sbatch --export)}"
: "${STEPS:?STEPS must be set (exported via sbatch --export)}"
: "${OUT:?OUT must be set (exported via sbatch --export)}"

echo "Host: $(hostname)"
echo "run_dir=${RUN_DIR} steps=${STEPS} out=${OUT}"

python scripts/spectral_from_checkpoints.py "${RUN_DIR}" --steps "${STEPS}" --out "${OUT}"
