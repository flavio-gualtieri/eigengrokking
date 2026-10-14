#!/bin/bash
# CPU smoke test of the Phase 1 pilot: seed 0 of each arm of
# scripts/run_collapse_pilot_array.sh (same script, same config), cut to 200
# steps with a log point every 20, then checks every run's outputs.
#
# Usage: bash scripts/smoke_collapse_pilot.sh [out_dir]   (default: a new temporary directory)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT="${1:-$(mktemp -d)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" CUDA_VISIBLE_DEVICES=""

for TASK_ID in 0 10 20 30; do
    TASK_ID="${TASK_ID}" PILOT_OUTPUT_DIR="${OUT}" \
        EXTRA_OVERRIDES="optimization_steps=200 log_every=20 eval_every=20" \
        bash scripts/run_collapse_pilot_array.sh
done

python - "${OUT}" <<'EOF'
import json, sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
expected = {  # arm -> (adam_beta2, grad_clip, dtype)
    "A_default": (0.999, None, "float32"),
    "B_beta2_0.98": (0.98, None, "float32"),
    "C_beta2_0.98_clip1": (0.98, 1.0, "float32"),
    "D_float64": (0.999, None, "float64"),
}
diagnostics = ("train_losses", "grad_norms", "update_norms", "max_abs_logits", "frac_saturated", "last_layer_norms")
print(f"\n{'arm':20s} {'beta2':>6s} {'clip':>5s} {'dtype':>8s} {'logs':>5s}  " + "  ".join(f"{k:>14s}" for k in diagnostics))
for arm, (beta2, clip, dtype) in expected.items():
    (run,) = (out / arm).rglob("config_full.json")
    cfg = json.loads(run.read_text())
    assert (cfg["adam_beta2"], cfg["grad_clip"], cfg["dtype"]) == (beta2, clip, dtype), (arm, cfg)
    assert not any((run.parent / "checkpoints").iterdir()), f"{arm}: checkpoints were written"
    (npz,) = run.parent.glob("figures/training_data_*.npz")
    z = np.load(npz)
    n = len(z["train_steps"])
    for key in diagnostics:
        assert len(z[key]) == n and np.isfinite(z[key]).all(), (arm, key)
    assert ((0 <= z["frac_saturated"]) & (z["frac_saturated"] <= 1)).all() and (z["update_norms"] > 0).all(), arm
    print(f"{arm:20s} {beta2:6g} {str(clip):>5s} {dtype:>8s} {n:5d}  "
          + "  ".join(f"{z[k][-1]:14.4g}" for k in diagnostics))
print(f"\nSmoke test passed: 4 arms in {out}")
EOF
