# scripts/slurm_env.sh -- sourced (from the repo root) by every SLURM job
# script in scripts/.
#
# A job inherits the environment it was submitted from (sbatch exports it by
# default), so activating the project's env before `sbatch` is enough. To
# pin a specific virtualenv instead, export EIGENGROK_VENV=/path/to/venv
# before submitting.
if [[ -n "${EIGENGROK_VENV:-}" ]]; then
    source "${EIGENGROK_VENV}/bin/activate"
fi

if ! python -c "import eigengrok.train.loop" 2>/dev/null; then  # pulls in torch etc., not just the source tree
    echo "eigengrok is not importable from $(command -v python): activate its env or set EIGENGROK_VENV (see README)." >&2
    exit 1
fi
