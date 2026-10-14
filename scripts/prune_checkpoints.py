#!/usr/bin/env python3
# scripts/prune_checkpoints.py

"""
After a run finishes, deletes every checkpoint Phase 2 won't use: keeps the
log-spaced grid and the window around the run's own onset
(eigengrok/analysis/checkpoint_grid.py), with t_onset read off the run's
training curves. Writes checkpoints/kept_steps.json recording what was kept
and why.

Only touches runs trained with checkpoint_model_only=true (the sweep rerun,
configs/mod91_wd_sweep_beta2_0.98.yaml), so the original sweep's full
checkpoints can never be pruned by mistake. A run that never reaches onset
keeps all its checkpoints.

Usage (the same --config/--set as python -m eigengrok.train, or a run directory):
    python scripts/prune_checkpoints.py --config configs/mod91_wd_sweep_beta2_0.98.yaml --set weight_decay=0.6 seed=3
    python scripts/prune_checkpoints.py --run-dir <run_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eigengrok.analysis.checkpoint_grid import PER_DECADE, WINDOW_HI, WINDOW_LO, phase2_steps
from eigengrok.analysis.milestones import t_onset
from eigengrok.config import ExperimentConfig
from eigengrok.outputs import run_dir as config_run_dir

CKPT_RE = re.compile(r"checkpoint_step(\d+)\.pt$")


def prune(run: Path, *, dry_run: bool = False) -> dict:
    cfg = json.loads((run / "config_full.json").read_text())
    if not cfg.get("checkpoint_model_only", False):
        raise SystemExit(f"Refusing to prune {run}: not trained with checkpoint_model_only=true.")

    (npz_path,) = run.glob("figures/training_data_*.npz")
    with np.load(npz_path) as z:
        onset = t_onset(z["test_steps"], z["test_accuracies"])

    ckpts = {int(m.group(1)): p for p in (run / "checkpoints").glob("checkpoint_step*.pt")
             if (m := CKPT_RE.search(p.name))}
    if np.isfinite(onset):
        keep = set(phase2_steps(ckpts, onset))
    else:
        print(f"{run}: never reaches t_onset; keeping all {len(ckpts)} checkpoints")
        keep = set(ckpts)
    drop = sorted(set(ckpts) - keep)

    freed = sum(ckpts[s].stat().st_size for s in drop)
    if not dry_run:
        for s in drop:
            ckpts[s].unlink()
        record = {"t_onset": onset if np.isfinite(onset) else None, "per_decade": PER_DECADE,
                  "window": [WINDOW_LO, WINDOW_HI], "kept_steps": sorted(keep), "n_deleted": len(drop)}
        (run / "checkpoints" / "kept_steps.json").write_text(json.dumps(record, indent=1))
    print(f"{run}: t_onset={onset:g}, kept {len(keep)} of {len(ckpts)} checkpoints, "
          f"{'would free' if dry_run else 'freed'} {freed / 2**20:.1f} MiB")
    return {"kept": sorted(keep), "dropped": drop}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--config", type=Path, help="YAML config the run was trained with")
    where.add_argument("--run-dir", type=Path)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted, delete nothing")
    args = parser.parse_args()

    run = args.run_dir or config_run_dir(ExperimentConfig.from_yaml(args.config, overrides=args.overrides))
    prune(run, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
