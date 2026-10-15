#!/bin/bash
# CPU smoke test of scripts/run_spectral_sweep_array.sh on the runs written by
# scripts/smoke_wd_sweep_rerun.sh (tasks 0 and 239), with tiny Lanczos settings.
#
# Usage: bash scripts/smoke_spectral_sweep.sh <rerun_smoke_out_dir> [spectra_out_dir]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNS="${1:?pass the out_dir given to scripts/smoke_wd_sweep_rerun.sh}"
OUT="${2:-$(mktemp -d)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" CUDA_VISIBLE_DEVICES=""

for TASK_ID in 0 239; do
    TASK_ID="${TASK_ID}" DEVICE=cpu SPECTRAL_M=5 SPECTRAL_K=2 SPECTRA_OUT_DIR="${OUT}" \
        EXTRA_OVERRIDES="output_dir=${RUNS}" bash scripts/run_spectral_sweep_array.sh
done

python - "${OUT}" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
files = sorted(Path(sys.argv[1]).glob("*.parquet"))
assert [f.name for f in files] == ["MODULAR_p91_init=1_wd=0.45_seed=0.parquet", "MODULAR_p91_init=8_wd=1_seed=9.parquet"], files
for f in files:
    d = pd.read_parquet(f)
    for col in ("top_eig", "trace", "effective_rank", "gn_top_eig", "gn_trace", "gn_effective_rank", "seconds"):
        assert np.isfinite(d[col]).all(), (f.name, col)
    print(f"{f.name}: {len(d)} checkpoints, {d.shape[1]} columns")
print("Smoke test passed")
PY
