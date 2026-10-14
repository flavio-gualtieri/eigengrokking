#!/bin/bash
# Submits the multi-seed spectral-extraction replication: seeds 1-9 at
# wd=0.6 and wd=1 (seed=0 already exists as
# reports/spectra/MODULAR_p91_wd={0.6,1}_seed=0.parquet), the two
# conditions with a measured effective_rank collapse
# (reports/figures/gate_p91_*_seed0_summary.md: 13.4x at wd=0.6, 14.2x at wd=1).
#
# Step windows match what seed=0 already used (reports/spectra/README.md):
# wd=0.6 -> 100:6600:100 (66 checkpoints, onset ~3000); wd=1 ->
# 100:15000:100 (150 checkpoints, onset ~2400). Checked against every
# seed's own accuracy-derived onset (reports/wd_sweep_accuracy_curves.csv,
# first crossing of 2x chance): onset ranges 100-4400 at wd=0.6 and
# 100-2800 at wd=1 across seeds, so both windows bracket that (secondary)
# onset for every seed. They do NOT all bracket the primary onset (first
# test accuracy >= 0.5, reports/sweep_milestones.csv): at wd=0.6 seeds 1, 5,
# 6 and 7 reach it at steps 6800-9100, after the 6600 window ends.
#
# Time budgets are sized from seed=0's actual run time (30 min at wd=0.6,
# ~1h50 at wd=1), not the 59-minute default in
# run_spectral_from_checkpoints.sh, which timed out on the wd=1 case before
# -- see memory eigengrokking-slurm-jobs: raise wall time rather than trim
# resolution to fit a budget nobody asked for.
#
# Prerequisite: wd=0.6 seeds 1-9 are missing config_full.json. Run
# `python scripts/backfill_config_full.py` first, or these array tasks
# will exit immediately (checked inside run_spectral_wd_sweep_array.sh).
#
# Usage: ./scripts/submit_spectral_multiseed.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARRAY_SCRIPT="scripts/run_spectral_wd_sweep_array.sh"

echo "Submitting wd=0.6 array (seeds 1-9)"
sbatch --array=1-9 --time=1:30:00 --export="ALL,WD=0.6,STEPS=100:6600:100" "${ARRAY_SCRIPT}"

echo "Submitting wd=1 array (seeds 1-9)"
sbatch --array=1-9 --time=3:00:00 --export="ALL,WD=1,STEPS=100:15000:100" "${ARRAY_SCRIPT}"
