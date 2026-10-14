#!/bin/bash
# Phase 1 pilot (configs/collapse_pilot.yaml): 4 arms x 10 seeds at p=91,
# wd=1.0, init=1, 30k steps, one GPU per array task. Task t runs arm t / 10
# with seed t % 10:
#
#   0-9    A_default           AdamW betas (0.9, 0.999), no clipping, float32 (the sweep's settings)
#   10-19  B_beta2_0.98        betas (0.9, 0.98)
#   20-29  C_beta2_0.98_clip1  betas (0.9, 0.98), global gradient-norm clipping at 1.0
#   30-39  D_float64           the sweep's settings in float64
#
# Usage, from the repo root with the env activated (or EIGENGROK_VENV set):
#   mkdir -p logs && sbatch scripts/run_collapse_pilot_array.sh
#
# Each run lands in $PILOT_OUTPUT_DIR/<arm>/MODULAR/no_spectrum/modulus=91/
# depth=1/width=128/init=1/wd=1/seed=<seed>/ (default runs/collapse_pilot).
# Also runs outside SLURM, which scripts/smoke_collapse_pilot.sh uses:
#   TASK_ID=<t> EXTRA_OVERRIDES="optimization_steps=200" bash scripts/run_collapse_pilot_array.sh

#SBATCH -J collapse_pilot
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --array=0-39
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 0:59:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

source scripts/slurm_env.sh

ARMS=(A_default B_beta2_0.98 C_beta2_0.98_clip1 D_float64)
declare -A ARM_OVERRIDES=(
    [A_default]=""
    [B_beta2_0.98]="adam_beta2=0.98"
    [C_beta2_0.98_clip1]="adam_beta2=0.98 grad_clip=1.0"
    [D_float64]="dtype=float64"
)
N_SEEDS=10

TASK="${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set TASK_ID when running outside SLURM}}"
if (( TASK < 0 || TASK >= ${#ARMS[@]} * N_SEEDS )); then
    echo "Task ${TASK} out of range 0-$(( ${#ARMS[@]} * N_SEEDS - 1 ))" >&2
    exit 1
fi
ARM="${ARMS[$(( TASK / N_SEEDS ))]}"
SEED=$(( TASK % N_SEEDS ))
OUT="${PILOT_OUTPUT_DIR:-runs/collapse_pilot}/${ARM}"

echo "Host: $(hostname)"
echo "task=${TASK} arm=${ARM} seed=${SEED} out=${OUT} overrides=${ARM_OVERRIDES[$ARM]} ${EXTRA_OVERRIDES:-}"

# The override strings are deliberately word-split into separate key=value arguments.
# shellcheck disable=SC2086
python -m eigengrok.train --config configs/collapse_pilot.yaml --set \
    seed="${SEED}" output_dir="${OUT}" ${ARM_OVERRIDES[$ARM]} ${EXTRA_OVERRIDES:-}
