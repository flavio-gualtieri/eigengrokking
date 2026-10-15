#!/bin/bash
# Phase 2: Hessian (H) and Gauss-Newton (G) spectral statistics for every
# run of the sweep rerun (configs/mod91_wd_sweep_beta2_0.98.yaml), at every
# checkpoint the run kept (scripts/prune_checkpoints.py: a log-spaced grid
# plus every 100 steps within 0.5-1.5 x the run's onset). Task t covers the
# same run as task t of scripts/run_wd_sweep_rerun_array.sh, so it can start
# as soon as that run has finished:
#
#   mkdir -p logs && sbatch --dependency=aftercorr:<rerun job id> scripts/run_spectral_sweep_array.sh
#
# Output: one parquet per run, $SPECTRA_OUT_DIR/MODULAR_p91_init=<init>_wd=<wd>_seed=<seed>.parquet
# (default reports/spectra_sweep/; columns in scripts/spectral_from_checkpoints.py).
# SPECTRAL_M / SPECTRAL_K default to the setting chosen in reports/spectral_validation.md.
# Also runs outside SLURM (TASK_ID=<t> DEVICE=cpu ...), see scripts/smoke_spectral_sweep.sh.

#SBATCH -J spectral_sweep
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --array=0-239
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:00:0
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

# The run directory and output name, from the same code that laid the run out (eigengrok/outputs.py).
# shellcheck disable=SC2086
read -r RUN_DIR NAME < <(python - "${CONFIG}" weight_decay="${WD}" initialization_scale="${INIT}" seed="${SEED}" ${EXTRA_OVERRIDES:-} <<'PY'
import sys
from eigengrok.config import ExperimentConfig
from eigengrok.outputs import _fmt_float, run_dir
cfg = ExperimentConfig.from_yaml(sys.argv[1], overrides=sys.argv[2:])
print(run_dir(cfg), f"MODULAR_p{cfg.modulus}_init={_fmt_float(cfg.initialization_scale)}_wd={_fmt_float(cfg.weight_decay)}_seed={cfg.seed}")
PY
)
OUT="${SPECTRA_OUT_DIR:-reports/spectra_sweep}/${NAME}.parquet"

echo "Host: $(hostname)"
echo "task=${TASK} init=${INIT} wd=${WD} seed=${SEED} run_dir=${RUN_DIR} out=${OUT}"

python scripts/spectral_from_checkpoints.py "${RUN_DIR}" --steps all --operators hessian,gauss_newton \
    --spectral_m "${SPECTRAL_M:-20}" --spectral_k "${SPECTRAL_K:-20}" ${DEVICE:+--device "${DEVICE}"} --out "${OUT}"
