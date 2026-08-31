#!/bin/bash
# Fills the alpha=8 gap in the wd sweep: alpha=1 was run at wd in
# {0.45..1.00 step 0.05} (see submit_wd_sweep.sh), but alpha=8 was only run
# at wd>=0.70 (see reports/wd_sweep_accuracy_curves.csv and the
# fig:onset-wd caption in writeup.py). This submits just the missing
# low-wd end for alpha=8 so it matches alpha=1's range, without
# resubmitting (and burning GPU time re-running) the wd=0.70..1.00 points
# alpha=8 already has.
#
# Reuses run_wd_sweep_array.sh unchanged (it's already parameterized by the
# WEIGHT_DECAY/INIT_SCALE env vars) -- same array script submit_wd_sweep.sh
# uses, so results land in the same results/.../init=8/wd=<wd>/seed=<seed>
# layout build_wd_sweep_accuracy_csv.py expects.
#
# Usage: ./submit_wd_sweep_alpha8_remaining.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARRAY_SCRIPT="scripts/run_wd_sweep_array.sh"
INIT_SCALE=8

# The gap: alpha=1's swept range (submit_wd_sweep.sh) minus the wd>=0.70
# alpha=8 already has.
WEIGHT_DECAYS=(0.65 0.60 0.55 0.50 0.45)

for wd in "${WEIGHT_DECAYS[@]}"; do
    echo "Submitting array job for weight_decay=${wd} init_scale=${INIT_SCALE}"
    sbatch --job-name="wd${wd}_init${INIT_SCALE}" --export="ALL,WEIGHT_DECAY=${wd},INIT_SCALE=${INIT_SCALE}" "${ARRAY_SCRIPT}"
done
