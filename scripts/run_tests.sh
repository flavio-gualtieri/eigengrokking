#!/bin/bash

#SBATCH -J eigengrokking_tests
#SBATCH -p computeshort
#SBATCH -A pilot
#SBATCH -n 4
#SBATCH -t 0:20:0
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source scripts/slurm_env.sh

OMP_NUM_THREADS=1 python -m pytest -x -q "$@"
