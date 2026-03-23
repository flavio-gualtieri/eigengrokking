#!/bin/bash

#SBATCH -J eigengrokking_gpu_test
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --array=0-9
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 20:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs_gpu_test/slurm_%A_%a.out
#SBATCH --error=logs_gpu_test/slurm_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_gpu_test

module load miniforge
mamba activate /gpfs/scratch/qp252676/globus/grokking/grok-env

echo "Host: $(hostname)"
echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Task id: $SLURM_ARRAY_TASK_ID"
echo "Assigned GPU: ${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python main.py