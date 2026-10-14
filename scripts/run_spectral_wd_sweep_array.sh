#!/bin/bash
# Runs scripts/spectral_from_checkpoints.py as a SLURM array job, one task
# per seed, at a fixed weight_decay. Array counterpart to the single-run
# scripts/run_spectral_from_checkpoints.sh, for the multi-seed replication
# of the effective_rank collapse (reports/figures/gate_p91_*_seed0_summary.md): does the
# 13.4x / 14.2x drop seen at seed=0 hold across seeds, with error bars.
#
# Usage:
#   sbatch --array=1-9 --time=<budget> \
#       --export="ALL,WD=<weight_decay>,STEPS=<start:stop:stride>" \
#       scripts/run_spectral_wd_sweep_array.sh
#
# WD must match an existing $EIGENGROK_OUTPUT_DIR/.../wd=<WD>/ directory name exactly
# (EIGENGROK_OUTPUT_DIR defaults to results/, as in eigengrok/config.py)
# (e.g. "0.6", "1" -- not "1.0"). SLURM_ARRAY_TASK_ID is used as the seed;
# RUN_DIR is derived from WD + seed per eigengrok/outputs.py's layout,
# and OUT follows reports/spectra/README.md's naming convention. See
# scripts/submit_spectral_multiseed.sh for the two submissions this project
# needs (wd=0.6 and wd=1, seeds 1-9 -- seed=0 already exists).

#SBATCH -J spectral_multiseed
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 3:00:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%x_wd%A_%a.out
#SBATCH --error=logs/slurm_%x_wd%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs reports/spectra

source scripts/slurm_env.sh

: "${WD:?WD must be set (exported via sbatch --export)}"
: "${STEPS:?STEPS must be set (exported via sbatch --export)}"

SEED="${SLURM_ARRAY_TASK_ID}"
RUN_DIR="${EIGENGROK_OUTPUT_DIR:-results}/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd=${WD}/seed=${SEED}"
OUT="reports/spectra/MODULAR_p91_wd=${WD}_seed=${SEED}.parquet"

if [ ! -f "${RUN_DIR}/config_full.json" ]; then
    echo "Missing ${RUN_DIR}/config_full.json -- run scripts/backfill_config_full.py first" >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "wd=${WD} seed=${SEED} run_dir=${RUN_DIR} steps=${STEPS} out=${OUT}"

python scripts/spectral_from_checkpoints.py "${RUN_DIR}" --steps "${STEPS}" --out "${OUT}"
