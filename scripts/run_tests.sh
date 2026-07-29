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

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/grok-env
set -u

python -c "import pytest" 2>/dev/null || pip install --quiet pytest

python -m pytest -x -q "$@"
