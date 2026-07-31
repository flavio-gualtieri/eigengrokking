#!/bin/bash

#SBATCH -J eigengrokking_wd_sweep
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --array=0-9
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 0:59:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_wd%x_%A_%a.out
#SBATCH --error=logs/slurm_wd%x_%A_%a.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/grok-env
set -u

: "${WEIGHT_DECAY:?WEIGHT_DECAY must be set (exported via sbatch --export)}"
INIT_SCALE="${INIT_SCALE:-1.0}"

echo "Host: $(hostname)"
echo "weight_decay=${WEIGHT_DECAY} init_scale=${INIT_SCALE} seed=${SLURM_ARRAY_TASK_ID}"

python scripts/run_grok_mod91.py \
    --weight_decay "${WEIGHT_DECAY}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --optimization_steps 150000 \
    --initialization_scale "${INIT_SCALE}"
