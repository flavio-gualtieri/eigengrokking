#!/usr/bin/env python3
# analysis/effective_rank_compare.py

"""
Effective-rank collapse across weight decay: the one spectral quantity that
agrees between the two real single-run gates currently on disk (see
analysis/make_figures.py; Section "Current results and their status" of
writeup.py).

Aligns each run on its own onset step (2x-chance crossing, from
analysis/transitions.py::find_onset) and plots effective_rank_rel and
top_corr_rel on a shared "steps since onset" axis, so the post-onset
collapse can be compared directly across runs regardless of where onset
itself falls in absolute step count.

Regenerates entirely from committed parquets -- no notebook, one command:

    python -m analysis.effective_rank_compare

Defaults to the two runs that are actually on disk today (wd=0.6, wd=1,
both seed 0); pass --run to compare others as more spectral parquets get
produced (see scripts/spectral_from_checkpoints.py).

Output (see --out-dir, default reports/figures/):
    effective_rank_compare.png / .pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from analysis.make_figures import COLOR_EFF_RANK, COLOR_ONSET, COLOR_TOP_CORR, _mark_low_train_acc
from analysis.transitions import (
    add_corrected_quantities,
    find_onset,
    load_and_join,
    load_modulus,
    load_train_accuracy,
)

DEFAULT_RUNS: List[Tuple[str, str, str]] = [
    (
        "reports/spectra/MODULAR_p91_wd=0.6_seed=0.parquet",
        "results/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd=0.6/seed=0",
        "wd=0.6",
    ),
    (
        "reports/spectra/MODULAR_p91_wd=1_seed=0.parquet",
        "results/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd=1/seed=0",
        "wd=1",
    ),
]

# One linestyle per run, so quantity keeps its color identity (as in
# analysis/make_figures.py) and run identity rides on linestyle instead.
LINESTYLES = ["-", "--", "-.", ":"]


@dataclass
class RunSeries:
    label: str
    df: pd.DataFrame  # carries step, step_since_onset, *_rel, train_acc_low
    onset_step: int


def _load_run(parquet: Path, run_dir: Path, label: str) -> RunSeries:
    modulus = load_modulus(run_dir)
    df = load_and_join(parquet, run_dir)
    df = load_train_accuracy(df, run_dir)  # flags train_acc_low, same threshold as make_figures.py
    onset = find_onset(df, modulus)
    df = add_corrected_quantities(df)
    df["step_since_onset"] = df["step"] - onset.cross_step
    return RunSeries(label=f"{label} (onset step {onset.cross_step})", df=df, onset_step=onset.cross_step)


def plot_comparison(runs: List[RunSeries]) -> plt.Figure:
    fig, (ax_rank, ax_top) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    thresh = None
    for series, ls in zip(runs, LINESTYLES):
        df = series.df
        thresh = df.attrs.get("train_acc_low_thresh", thresh)
        ax_rank.plot(df["step_since_onset"], df["effective_rank_rel"], color=COLOR_EFF_RANK,
                     linestyle=ls, linewidth=1.5, alpha=0.8, label=series.label)
        ax_top.plot(df["step_since_onset"], df["top_corr_rel"], color=COLOR_TOP_CORR,
                    linestyle=ls, linewidth=1.5, alpha=0.8, label=series.label)
        # Hollow markers for checkpoints an AdamW/weight-decay dip in train
        # accuracy makes unreliable (see analysis/transitions.py::load_train_accuracy)
        # -- the raw step-to-step series is noisy even away from these points,
        # so this flags *part* of the noise, not all of it.
        _mark_low_train_acc(ax_rank, df.assign(step=df["step_since_onset"]), "effective_rank_rel", COLOR_EFF_RANK)
        _mark_low_train_acc(ax_top, df.assign(step=df["step_since_onset"]), "top_corr_rel", COLOR_TOP_CORR)

    low_label = f"train_acc < {thresh:g} (unreliable checkpoint)" if thresh is not None else None
    for ax, ylabel in ((ax_rank, "effective_rank_rel"), (ax_top, "top_eig_corr_rel")):
        ax.axvline(0, color=COLOR_ONSET, linewidth=1, alpha=0.8, label="onset" if ax is ax_rank else None)
        ax.axhline(1.0, color="black", linewidth=0.75, alpha=0.4)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
    if low_label:
        ax_rank.scatter([], [], facecolors="none", edgecolors="black", marker="o", label=low_label)
    ax_rank.legend(loc="best", fontsize=8)
    ax_top.legend(loc="best", fontsize=8)

    ax_top.set_xlabel("Steps since onset (2x-chance crossing)")
    ax_rank.set_title("Effective-rank collapse aligns across weight decay; top-eig rise is looser")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=3, action="append", metavar=("PARQUET", "RUN_DIR", "LABEL"),
                         help="Repeatable. Defaults to the two wd=0.6/wd=1 runs on disk if omitted.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--tag", default="effective_rank_compare")
    args = parser.parse_args()

    raw_runs = args.run if args.run else DEFAULT_RUNS
    runs = [_load_run(Path(p), Path(r), l) for p, r, l in raw_runs]

    fig = plot_comparison(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"{args.tag}.png"
    pdf_path = args.out_dir / f"{args.tag}.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
