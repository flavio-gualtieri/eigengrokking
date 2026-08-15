#!/bin/bash
# Runs analysis/make_figures.py (the gate figure) as a single CPU SLURM job.
# No GPU needed -- this only reads a committed spectral parquet plus the
# run's training-curve npz (see reports/spectra/README.md).
#
# Usage:
#   sbatch --export="ALL,PARQUET=<parquet path>,RUN_DIR=<run base dir>" \
#       scripts/run_make_figures.sh
#
# RUN_DIR must be a run's base directory (contains checkpoints/,
# config_full.json, figures/training_data_*.npz -- see
# project_io/dir_making.py). PARQUET is produced by
# scripts/spectral_from_checkpoints.py (see scripts/run_spectral_from_checkpoints.sh).

#SBATCH -J make_figures
#SBATCH -p computeshort
#SBATCH -A pilot
#SBATCH -n 4
#SBATCH -t 0:30:0
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/grok-env
set -u

: "${PARQUET:?PARQUET must be set (exported via sbatch --export)}"
: "${RUN_DIR:?RUN_DIR must be set (exported via sbatch --export)}"

echo "Host: $(hostname)"
echo "parquet=${PARQUET} run_dir=${RUN_DIR}"

python -m analysis.make_figures "${PARQUET}" "${RUN_DIR}"
