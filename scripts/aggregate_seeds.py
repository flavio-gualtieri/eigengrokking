#!/usr/bin/env python3
# scripts/aggregate_seeds.py

"""
Aggregates a seed sweep (see configs/sweeps.py) into mean +/- std curves.

Usage:
    python scripts/aggregate_seeds.py <family_dir>

<family_dir> is the directory that directly contains the `seed=N/` run
directories, e.g.:
    results/MODULAR/depth=1/width=128/init=1/wd=1

Single-seed grokking curves (and single-seed spectral statistics) are
notoriously variable; this pools >=5 seeds of the same architecture/
hyperparameters into a mean +/- std band, complementing the per-run,
per-checkpoint +/-stderr-over-probes that training/loop.py already saves.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.spectral_observables import SpectralObservables  # noqa: E402

ACCURACY_KEYS = [
    ("train_steps", "train_accuracies", "Train"),
    ("test_steps", "test_accuracies", "Test"),
]

# Every observable training/loop.py logs per checkpoint (see SpectralObservables).
SPECTRAL_KEYS = [f.name for f in fields(SpectralObservables)]


def _load_runs(family_dir: Path) -> List[Dict[str, np.ndarray]]:
    npz_paths = sorted(family_dir.glob("seed=*/figures/training_data_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No training_data_*.npz found under {family_dir}/seed=*/figures/")

    runs = []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as data:
            runs.append({k: data[k] for k in data.files})
    return runs


def _interp_onto_grid(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(steps)
    return np.interp(grid, steps[order], values[order])


def _aggregate_on_common_grid(
        runs: List[Dict[str, np.ndarray]],
        step_key: str,
        value_key: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolates each run's (step_key, value_key) series onto a shared grid
    spanning the intersection of all runs' step ranges, then returns
    (grid, mean, std) across runs.

    Interpolation (rather than a step-exact join) matters for the spectral
    observables specifically: the adaptive logging schedule
    (training/spectral_schedule.py) reacts to each run's own accuracy
    trajectory, so different seeds generally snapshot the spectrum at
    different steps.
    """
    per_run = [
        (r[step_key].astype(float), r[value_key].astype(float))
        for r in runs
        if step_key in r and value_key in r and r[step_key].size > 0
    ]
    if len(per_run) < 2:
        raise ValueError(f"Need >=2 runs with '{value_key}' data to aggregate; got {len(per_run)}.")

    lo = max(steps.min() for steps, _ in per_run)
    hi = min(steps.max() for steps, _ in per_run)
    if hi <= lo:
        raise ValueError(f"No overlapping step range across runs for '{value_key}'.")

    n_points = max(len(steps) for steps, _ in per_run)
    grid = np.linspace(lo, hi, n_points)

    stacked = np.stack([_interp_onto_grid(steps, values, grid) for steps, values in per_run], axis=0)
    return grid, stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


def _plot_band(
        path: Path,
        grid: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        ylabel: str,
        title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, mean, color="purple", label="mean")
    ax.fill_between(grid, mean - std, mean + std, color="purple", alpha=0.2, linewidth=0, label="+/-1 std")
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def aggregate(family_dir: Path, out_dir: Optional[Path] = None) -> None:
    runs = _load_runs(family_dir)
    n_seeds = len(runs)
    print(f"Aggregating {n_seeds} seeds from {family_dir}")
    if n_seeds < 5:
        print(f"WARNING: only {n_seeds} seeds found; error bars will be noisy with < 5.")

    out_dir = out_dir or (family_dir / "seed_aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, np.ndarray] = {}

    for step_key, value_key, label in ACCURACY_KEYS:
        try:
            grid, mean, std = _aggregate_on_common_grid(runs, step_key, value_key)
        except ValueError as e:
            print(f"Skipping {value_key}: {e}")
            continue
        saved[f"{value_key}_grid"] = grid
        saved[f"{value_key}_mean"] = mean
        saved[f"{value_key}_std"] = std
        _plot_band(
            out_dir / f"{value_key}_aggregate.png", grid, mean, std,
            ylabel=f"{label} accuracy",
            title=f"{label} accuracy: mean +/- std over {n_seeds} seeds",
        )

    for key in SPECTRAL_KEYS:
        try:
            grid, mean, std = _aggregate_on_common_grid(runs, "eig_steps", key)
        except ValueError as e:
            print(f"Skipping {key}: {e}")
            continue
        saved[f"{key}_grid"] = grid
        saved[f"{key}_mean"] = mean
        saved[f"{key}_std"] = std
        _plot_band(
            out_dir / f"{key}_aggregate.png", grid, mean, std,
            ylabel=key,
            title=f"{key}: mean +/- std over {n_seeds} seeds",
        )

    np.savez_compressed(out_dir / "seed_aggregate.npz", **saved)
    print(f"Wrote aggregate plots/data to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family_dir", type=Path, help="Directory containing seed=N/ run subdirectories")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    aggregate(args.family_dir, args.out_dir)


if __name__ == "__main__":
    main()
