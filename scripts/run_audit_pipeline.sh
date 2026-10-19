#!/bin/bash
# Everything after the GPU jobs, on one CPU node: extract the rerun's curves,
# summarise the rerun and compare it with the original sweep, run the
# forecasting audit (with the spectra), time each indicator's change against
# the transition, draw the figures, and regenerate
# paper/numbers.tex and RESULTS.md. Writes into the repo's reports/,
# paper/ and RESULTS.md; commits nothing.
#
# Queue it to start when the spectra job finishes (every task, success or not):
#   sbatch --dependency=afterany:<spectra job id> --export="ALL,EIGENGROK_REPO=<repo>" scripts/run_audit_pipeline.sh
# Submitted from a frozen copy of the code (its directory is the code that runs;
# EIGENGROK_REPO is where the data and outputs live). Locally:
#   EIGENGROK_REPO=$PWD bash scripts/run_audit_pipeline.sh

#SBATCH -J audit_pipeline
#SBATCH -p compute
#SBATCH -A pilot
#SBATCH -n 4
#SBATCH -t 4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail

CODE="${SLURM_SUBMIT_DIR:-$PWD}"
REPO="${EIGENGROK_REPO:?set EIGENGROK_REPO to the repo whose data and reports to update}"
export PYTHONPATH="${CODE}" EIGENGROK_REPO="${REPO}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "${CODE}" && source scripts/slurm_env.sh
cd "${REPO}"

echo "Host: $(hostname)  code: ${CODE}  repo: ${REPO}  start: $(date)"
failed=()
step() {
    local name="$1"; shift
    echo "== ${name}  ($(date +%T))"
    if ! python "${CODE}/scripts/$@"; then
        echo "!! ${name} FAILED"
        failed+=("${name}")
    fi
}

step "extract curves" build_wd_sweep_accuracy_csv.py results/sweep_beta2_0.98 \
    --out reports/sweep_beta2_0.98_accuracy_curves.csv.gz --norms-out reports/sweep_beta2_0.98_training_stats.csv.gz
step "rerun summary" sweep_summary.py --accuracy-csv reports/sweep_beta2_0.98_accuracy_curves.csv.gz --out-dir reports/sweep_beta2_0.98
step "compare sweeps" compare_sweeps.py
step "forecast audit" forecast_benchmark.py
step "transition lead times" transition_leadtimes.py
step "figures" figures/forecast_figures.py
step "numbers" make_numbers.py

echo "spectra files: $(ls reports/spectra_sweep/*.parquet 2>/dev/null | wc -l)"
echo "done: $(date)  failed steps: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
