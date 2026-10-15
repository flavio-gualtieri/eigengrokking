#!/bin/bash
# GPU benchmark of the post-hoc spectral cost per checkpoint over the candidate
# (m, k) settings (scripts/bench_spectral_settings.py -> reports/spectral_benchmark.md).
#
# Usage, from the repo root with the env activated (or EIGENGROK_VENV set):
#   mkdir -p logs && sbatch scripts/run_bench_spectral.sh

#SBATCH -J bench_spectral
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 0:30:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/slurm_env.sh
echo "Host: $(hostname)"
python scripts/bench_spectral_settings.py --out "${BENCH_OUT:-reports/spectral_benchmark.md}"
