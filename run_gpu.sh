#!/bin/bash

#SBATCH -J eigengrokking_gpu_test
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

module load miniforge
mamba activate /gpfs/scratch/qp252676/globus/grokking/grok-env

echo "Host: $(hostname)"
echo "Assigned GPU: ${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python main.py
