#!/bin/bash
# The p=91 sweep rerun (configs/mod91_wd_sweep_beta2_0.98.yaml): 2 init scales
# x 12 weight decays x 10 seeds, one GPU per array task. Task t runs
#   init = INITS[t / 120], weight decay = WDS[(t / 10) % 12], seed = t % 10,
# then prunes that run's checkpoints to the Phase 2 grid (scripts/prune_checkpoints.py).
#
# Usage, from the repo root with the env activated (or EIGENGROK_VENV set):
#   mkdir -p logs && sbatch scripts/run_wd_sweep_rerun_array.sh
#
# Runs land in results/sweep_beta2_0.98/MODULAR/no_spectrum/modulus=91/.../seed=<seed>/.
# Also runs outside SLURM (scripts/smoke_wd_sweep_rerun.sh):
#   TASK_ID=<t> EXTRA_OVERRIDES="optimization_steps=300" bash scripts/run_wd_sweep_rerun_array.sh

#SBATCH -J wd_sweep_rerun
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --array=0-239
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:30:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

source scripts/slurm_env.sh

CONFIG="${CONFIG:-configs/mod91_wd_sweep_beta2_0.98.yaml}"
INITS=(1 8)
WDS=(0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
N_SEEDS=10
N_TASKS=$(( ${#INITS[@]} * ${#WDS[@]} * N_SEEDS ))

TASK="${SLURM_ARRAY_TASK_ID:-${TASK_ID:?set TASK_ID when running outside SLURM}}"
if (( TASK < 0 || TASK >= N_TASKS )); then
    echo "Task ${TASK} out of range 0-$(( N_TASKS - 1 ))" >&2
    exit 1
fi
INIT="${INITS[$(( TASK / (${#WDS[@]} * N_SEEDS) ))]}"
WD="${WDS[$(( (TASK / N_SEEDS) % ${#WDS[@]} ))]}"
SEED=$(( TASK % N_SEEDS ))
SET=(weight_decay="${WD}" initialization_scale="${INIT}" seed="${SEED}")

echo "Host: $(hostname)"
echo "task=${TASK} config=${CONFIG} init=${INIT} wd=${WD} seed=${SEED} extra=${EXTRA_OVERRIDES:-}"

# EXTRA_OVERRIDES is deliberately word-split into separate key=value arguments.
# shellcheck disable=SC2086
python -m eigengrok.train --config "${CONFIG}" --set "${SET[@]}" ${EXTRA_OVERRIDES:-}
# shellcheck disable=SC2086
python scripts/prune_checkpoints.py --config "${CONFIG}" --set "${SET[@]}" ${EXTRA_OVERRIDES:-}
