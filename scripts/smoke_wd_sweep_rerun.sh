#!/bin/bash
# CPU smoke test of the sweep rerun: the first and last task of
# scripts/run_wd_sweep_rerun_array.sh (init 1, wd 0.45, seed 0 and init 8,
# wd 1.0, seed 9), cut to 300 steps with checkpoints every 50, then the
# Phase 2 path on what's left: spectra from the kept checkpoints.
#
# Usage: bash scripts/smoke_wd_sweep_rerun.sh [out_dir]   (default: a new temporary directory)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT="${1:-$(mktemp -d)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" CUDA_VISIBLE_DEVICES=""
EXTRA="optimization_steps=300 log_every=50 eval_every=50 checkpoint_every=50 output_dir=${OUT}"

for TASK_ID in 0 239; do
    TASK_ID="${TASK_ID}" EXTRA_OVERRIDES="${EXTRA}" bash scripts/run_wd_sweep_rerun_array.sh
done

for RUN in "${OUT}"/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=*/wd=*/seed=*; do
    python scripts/spectral_from_checkpoints.py "${RUN}" --steps 1,50,100 --out "${RUN}/smoke_spectra.parquet" \
        --spectral_m 5 --spectral_k 2 --device cpu
done
echo "Smoke test passed: ${OUT}"
