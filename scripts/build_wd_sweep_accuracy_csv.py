#!/usr/bin/env python3
# scripts/build_wd_sweep_accuracy_csv.py

"""
Flattens every run's logged training curves under results/ into two
long-format CSVs, one row per (run, logged step):

    --out        train/test accuracy and loss       (reports/wd_sweep_accuracy_curves.csv)
    --norms-out  total weight norm, last-layer norm, and where logged:
                 gradient norm, update norm, max |logit|,
                 softmax-saturated fraction          (reports/wd_sweep_norms.csv)

`results/` itself is gitignored (raw per-run checkpoints/figures, ~tens of
GB) -- this script is how the wd-sweep's actual signal gets preserved in the
repo without the raw run directories. modulus/depth/width/init/wd/seed
identify the run. Columns the training loop didn't log yet when a run was
trained (eigengrok/train/loop.py) are empty for that run.

Files under results/ have been disappearing (a scratch-filesystem purge), so
an output that already exists is never overwritten with fewer runs than it
holds: the script reports the runs it would lose and leaves that file
untouched, unless --allow-fewer-runs is given.

An output path ending in .gz is written gzip-compressed (with a fixed
timestamp, so identical data gives an identical file); pandas reads it as is.

Usage:
    python scripts/build_wd_sweep_accuracy_csv.py [results_dir] [--out CSV] [--norms-out CSV]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
from pathlib import Path

import numpy as np

RUN_DIR_RE = re.compile(
    r"modulus=(?P<modulus>[\d.]+)/depth=(?P<depth>[\d.]+)/width=(?P<width>[\d.]+)/"
    r"init=(?P<init>[\d.]+)/wd=(?P<wd>[\d.]+)/seed=(?P<seed>\d+)$"
)
RUN_COLUMNS = ["modulus", "depth", "width", "init", "wd", "seed"]
ACCURACY_COLUMNS = RUN_COLUMNS + ["step", "train_accuracy", "test_accuracy", "train_loss", "test_loss"]
OPTIONAL_KEYS = {"grad_norms": "grad_norm", "update_norms": "update_norm",  # npz key -> column; absent in older runs
                 "max_abs_logits": "max_abs_logit", "frac_saturated": "frac_saturated"}
NORM_COLUMNS = RUN_COLUMNS + ["step", "norm", "last_layer_norm", *OPTIONAL_KEYS.values()]


def _iter_runs(results_dir: Path):
    for npz_path in sorted(results_dir.glob("**/figures/training_data_*.npz")):
        run_dir = npz_path.parent.parent
        m = RUN_DIR_RE.search(str(run_dir.as_posix()))
        if not m:
            print(f"Skipping (unrecognized path shape): {run_dir}")
            continue
        yield m.groupdict(), npz_path


def _open(path: Path, mode: str):
    """Text-mode CSV handle; gzip for *.gz, with mtime 0 when writing so the bytes depend only on the data."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.GzipFile(path, mode + "b", mtime=0), newline="")
    return open(path, mode, newline="")


def _run_key(row) -> tuple:
    """Identifies a run independently of how its numbers were formatted ("1" vs "1.0")."""
    return tuple(float(row[c]) for c in RUN_COLUMNS)


def _f32(x: float) -> str:
    # The norms are logged from float32 tensors; 9 significant digits round-trip a float32 exactly.
    return f"{x:.9g}"


def collect(results_dir: Path) -> tuple[list[list], list[list]]:
    accuracy_rows, norm_rows = [], []
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
            norms = data["norms"]
            ll_norms = data["last_layer_norms"]
            optional = {col: data[key] for key, col in OPTIONAL_KEYS.items() if key in data.files and len(data[key])}

        run = [meta[c] for c in RUN_COLUMNS]
        for i in range(len(steps)):
            accuracy_rows.append(run + [
                int(steps[i]),
                float(train_acc[i]), float(test_acc[i]),
                float(train_loss[i]), float(test_loss[i]),
            ])

        aligned = len(norms) == len(ll_norms) == len(steps) and all(len(v) == len(steps) for v in optional.values())
        if not aligned:
            print(f"Skipping norms (not on the train_steps grid): {npz_path}")
            continue
        for i in range(len(steps)):
            norm_rows.append(run + [int(steps[i]), _f32(norms[i]), _f32(ll_norms[i])]
                             + ["" if col not in optional else _f32(optional[col][i]) for col in OPTIONAL_KEYS.values()])
    return accuracy_rows, norm_rows


def write_guarded(out_path: Path, header: list[str], rows: list[list], *, allow_fewer_runs: bool) -> bool:
    """Writes `rows` to `out_path` unless that would drop runs the existing file holds. True if written."""
    new_runs = {_run_key(dict(zip(header, r))) for r in rows}
    if out_path.exists() and not allow_fewer_runs:
        with _open(out_path, "r") as f:
            old_runs = {_run_key(r) for r in csv.DictReader(f)}
        lost = old_runs - new_runs
        if lost:
            print(f"NOT rewriting {out_path}: it holds {len(old_runs)} runs, {len(lost)} of which are no longer "
                  f"on disk (e.g. init/wd/seed {sorted(lost)[0][3:]}). Pass --allow-fewer-runs to overwrite anyway.")
            return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _open(out_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {len(new_runs)} runs ({len(rows)} rows) to {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("reports/wd_sweep_accuracy_curves.csv"))
    parser.add_argument("--norms-out", type=Path, default=Path("reports/wd_sweep_norms.csv"))
    parser.add_argument("--allow-fewer-runs", action="store_true",
                        help="Overwrite an existing output even if runs it holds are no longer under results_dir")
    args = parser.parse_args()

    accuracy_rows, norm_rows = collect(args.results_dir)
    write_guarded(args.out, ACCURACY_COLUMNS, accuracy_rows, allow_fewer_runs=args.allow_fewer_runs)
    write_guarded(args.norms_out, NORM_COLUMNS, norm_rows, allow_fewer_runs=args.allow_fewer_runs)


if __name__ == "__main__":
    main()
