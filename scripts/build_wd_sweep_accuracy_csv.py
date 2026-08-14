#!/usr/bin/env python3
# scripts/build_wd_sweep_accuracy_csv.py

"""
Flattens every run's accuracy curve under results/ into one long-format CSV.

`results/` itself is gitignored (raw per-run checkpoints/figures, ~tens of
GB) -- this script is how the wd-sweep's actual signal (train/test accuracy
over training) gets preserved in the repo without the raw run directories.
One row per (run, step): modulus/depth/width/init/wd/seed identify the run,
step/train_accuracy/test_accuracy/train_loss/test_loss carry the curve.

Usage:
    python scripts/build_wd_sweep_accuracy_csv.py [results_dir] [--out CSV]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

RUN_DIR_RE = re.compile(
    r"modulus=(?P<modulus>[\d.]+)/depth=(?P<depth>[\d.]+)/width=(?P<width>[\d.]+)/"
    r"init=(?P<init>[\d.]+)/wd=(?P<wd>[\d.]+)/seed=(?P<seed>\d+)$"
)


def _iter_runs(results_dir: Path):
    for npz_path in sorted(results_dir.glob("**/figures/training_data_*.npz")):
        run_dir = npz_path.parent.parent
        m = RUN_DIR_RE.search(str(run_dir.as_posix()))
        if not m:
            print(f"Skipping (unrecognized path shape): {run_dir}")
            continue
        yield m.groupdict(), npz_path


def build_csv(results_dir: Path, out_path: Path) -> int:
    n_runs = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "modulus", "depth", "width", "init", "wd", "seed",
            "step", "train_accuracy", "test_accuracy", "train_loss", "test_loss",
        ])

        for meta, npz_path in _iter_runs(results_dir):
            with np.load(npz_path, allow_pickle=True) as data:
                steps = data["train_steps"]
                if not np.array_equal(steps, data["test_steps"]):
                    print(f"Skipping (train/test step grids differ): {npz_path}")
                    continue
                train_acc = data["train_accuracies"]
                test_acc = data["test_accuracies"]
                train_loss = data["train_losses"]
                test_loss = data["test_losses"]

            for i in range(len(steps)):
                writer.writerow([
                    meta["modulus"], meta["depth"], meta["width"], meta["init"], meta["wd"], meta["seed"],
                    int(steps[i]),
                    float(train_acc[i]), float(test_acc[i]),
                    float(train_loss[i]), float(test_loss[i]),
                ])
            n_runs += 1

    return n_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("reports/wd_sweep_accuracy_curves.csv"))
    args = parser.parse_args()

    n_runs = build_csv(args.results_dir, args.out)
    print(f"Wrote {n_runs} runs' accuracy curves to {args.out}")


if __name__ == "__main__":
    main()
