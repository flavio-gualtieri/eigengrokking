#!/bin/bash
# Submits one separate SLURM array job per weight_decay value in the sweep.
# Each array job runs scripts/run_grok_mod91.py across seeds 0-9 (see
# run_wd_sweep_array.sh's --array range) at that fixed weight_decay.
#
# Usage: ./submit_wd_sweep.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARRAY_SCRIPT="scripts/run_wd_sweep_array.sh"

# Sweep from 1.0 down past 0.5, in steps of 0.05.
WEIGHT_DECAYS=(1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45)

for wd in "${WEIGHT_DECAYS[@]}"; do
    echo "Submitting array job for weight_decay=${wd}"
    sbatch --job-name="wd${wd}" --export="ALL,WEIGHT_DECAY=${wd}" "${ARRAY_SCRIPT}"
done
