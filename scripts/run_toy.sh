#!/bin/bash

#SBATCH -J eigengrokking_toy
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --array=0-9
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 20:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_toy

module load miniforge
mamba activate /gpfs/scratch/qp252676/globus/grokking/grok-env

export USE_TOY_MLP=1
python main.py
