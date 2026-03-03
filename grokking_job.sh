#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N grokking
#$ -t 1-5
#$ -pe smp 10
#$ -l h_rt=9:10:0
#$ -l h_vmem=4G
#$ -o outputs/

set -euo pipefail

: "${SGE_TASK_ID:?Run via qsub as an array job}"
EXP_INDEX=$((SGE_TASK_ID - 1))

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_PREFIX_ENV="${CONDA_PREFIX_ENV:-$PROJECT_DIR/env}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/runs}"
export OUTPUT_DIR

cd "$PROJECT_DIR"

# Make conda available in batch jobs
source /share/apps/rocky9/general/apps/miniforge/24.7.1/etc/profile.d/conda.sh

# Optional: ensure conda is really available
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda still not found"; exit 127; }

conda activate "$CONDA_PREFIX_ENV"

exec python main.py --exp_index "$EXP_INDEX"