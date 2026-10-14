#!/usr/bin/env python3
"""
Backfills config_full.json for wd=0.6 seeds 1-9.

That sweep predates eigengrok/outputs.py writing config_full.json at
training time -- only seed=0 was ever hand-patched (e.g. for the existing
gate figure), so seeds 1-9 have checkpoints/ but no config_full.json, which
scripts/spectral_from_checkpoints.py requires. All wd=0.6 seeds came from
the same sweep config (submit_wd_sweep.sh -> run_wd_sweep_array.sh), so
seed=0's config_full.json is correct for every other seed except the "seed"
field -- this just copies it with that one field overridden. Never touches
a file that already exists.

Run directly on the login node -- this is a local file write, not a
compute job. Usage: python scripts/backfill_config_full.py
"""
import json
from pathlib import Path

BASE = Path("results/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd=0.6")
TEMPLATE = BASE / "seed=0" / "config_full.json"


def main() -> None:
    template = json.loads(TEMPLATE.read_text())
    for seed in range(1, 10):
        seed_dir = BASE / f"seed={seed}"
        target = seed_dir / "config_full.json"
        if target.exists():
            print(f"seed={seed}: already present, skipping")
            continue
        if not (seed_dir / "checkpoints").is_dir():
            print(f"seed={seed}: no checkpoints/ dir, skipping")
            continue
        cfg = dict(template)
        cfg["seed"] = seed
        target.write_text(json.dumps(cfg, indent=2))
        print(f"seed={seed}: wrote {target}")


if __name__ == "__main__":
    main()
