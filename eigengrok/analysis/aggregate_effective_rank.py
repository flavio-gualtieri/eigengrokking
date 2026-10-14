#!/usr/bin/env python3
# eigengrok/analysis/aggregate_effective_rank.py

"""
Multi-seed replication of the effective_rank collapse (reports/figures/gate_p91_wd{0.6,1}_seed0_summary.md):
at seed=0, effective_rank fell 13.4x after onset at
wd=0.6 and 14.2x at wd=1 -- this was n=1 per condition. Now that
scripts/submit_spectral_multiseed.sh has produced seeds 1-9 at both wd,
this computes the same collapse factor (effective_rank_rel at onset /
effective_rank_rel at the window's last checkpoint, exactly
eigengrok/analysis/make_figures.py::write_summary's rank_div) per seed, and reports
mean +/- std per wd -- the direct test of whether the two-seed agreement
was real or a coincidence.

Regenerates entirely from committed parquets -- no notebook, one command:

    python -m eigengrok.analysis.aggregate_effective_rank

Outputs (see --out-dir, default reports/figures/):
    effective_rank_multiseed.png / .pdf   all seeds overlaid per wd, aligned
                                           to each run's own onset
    effective_rank_multiseed_summary.md   per-seed and mean +/- std collapse
                                           factors
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eigengrok.analysis.make_figures import COLOR_EFF_RANK, COLOR_ONSET, COLOR_TOP_CORR
from eigengrok.analysis.transitions import (
    add_corrected_quantities,
    find_onset,
    load_and_join,
    load_modulus,
    load_train_accuracy,
)

CONDITIONS: List[str] = ["0.6", "1"]
SEEDS: List[int] = list(range(10))


def _run_paths(wd: str, seed: int) -> tuple[Path, Path]:
    parquet = Path(f"reports/spectra/MODULAR_p91_wd={wd}_seed={seed}.parquet")
    results = Path(os.environ.get("EIGENGROK_OUTPUT_DIR", "results"))
    run_dir = results / f"MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd={wd}/seed={seed}"
    return parquet, run_dir


@dataclass
class SeedResult:
    seed: int
    onset_step: int  # 2x-chance crossing (secondary definition), what the collapse factor is measured from
    t_onset: float   # primary onset, first test_acc >= 0.5 (eigengrok/analysis/milestones.py), for comparison
    rank_div: float  # effective_rank_rel[onset] / effective_rank_rel[last] -- the collapse factor
    top_mult: float  # top_corr_rel[last] / top_corr_rel[onset]
    n_flagged: int  # checkpoints with train_acc_low in this window
    onset_flagged: bool
    last_flagged: bool
    df: pd.DataFrame  # carries step_since_onset, *_rel, train_acc_low, for the overlay figure


def _load_seed(wd: str, seed: int) -> SeedResult:
    parquet, run_dir = _run_paths(wd, seed)
    modulus = load_modulus(run_dir)
    df = load_and_join(parquet, run_dir)
    df = load_train_accuracy(df, run_dir)
    onset = find_onset(df, modulus)
    df = add_corrected_quantities(df)

    idx_onset = int((df["step"] - onset.cross_step).abs().idxmin())
    idx_last = len(df) - 1

    rank_div = float(df["effective_rank_rel"].iloc[idx_onset] / df["effective_rank_rel"].iloc[idx_last])
    top_mult = float(df["top_corr_rel"].iloc[idx_last] / df["top_corr_rel"].iloc[idx_onset])

    df["step_since_onset"] = df["step"] - onset.cross_step

    return SeedResult(
        seed=seed,
        onset_step=onset.cross_step,
        t_onset=onset.t_onset,
        rank_div=rank_div,
        top_mult=top_mult,
        n_flagged=int(df["train_acc_low"].sum()),
        onset_flagged=bool(df["train_acc_low"].iloc[idx_onset]),
        last_flagged=bool(df["train_acc_low"].iloc[idx_last]),
        df=df,
    )


def plot_overlay(results: Dict[str, List[SeedResult]]) -> plt.Figure:
    fig, axes = plt.subplots(2, len(results), figsize=(6.5 * len(results), 7), sharex="col")
    if len(results) == 1:
        axes = axes.reshape(2, 1)

    for col, (wd, seed_results) in enumerate(results.items()):
        ax_rank, ax_top = axes[0, col], axes[1, col]
        for r in seed_results:
            df = r.df
            ax_rank.plot(df["step_since_onset"], df["effective_rank_rel"], color=COLOR_EFF_RANK,
                         linewidth=1.0, alpha=0.35)
            ax_top.plot(df["step_since_onset"], df["top_corr_rel"], color=COLOR_TOP_CORR,
                        linewidth=1.0, alpha=0.35)
        for ax, ylabel in ((ax_rank, "effective_rank_rel"), (ax_top, "top_eig_corr_rel")):
            ax.axvline(0, color=COLOR_ONSET, linewidth=1, alpha=0.8)
            ax.axhline(1.0, color="black", linewidth=0.75, alpha=0.4)
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
        ax_rank.set_title(f"wd={wd} (n={len(seed_results)} seeds)")
        ax_top.set_xlabel("Steps since onset (2x-chance crossing)")

    fig.suptitle("Effective-rank collapse across seeds, aligned to each run's own onset")
    fig.tight_layout()
    return fig


def _step_or_note(step: float) -> str:
    return f"{step:.0f}" if np.isfinite(step) else "after window"


def write_summary(results: Dict[str, List[SeedResult]], out_path: Path) -> None:
    lines = [
        "# Multi-seed effective_rank collapse",
        "",
        "Generated by `python -m eigengrok.analysis.aggregate_effective_rank`. Collapse factor is "
        "`effective_rank_rel[onset] / effective_rank_rel[last checkpoint in window]`, the same "
        "definition `eigengrok/analysis/make_figures.py::write_summary` uses for the single-seed headline "
        "(so this is directly comparable to the 13.4x / 14.2x seed=0 numbers in reports/figures/gate_p91_wd{0.6,1}_seed0_summary.md).",
        "",
    ]
    for wd, seed_results in results.items():
        divs = np.array([r.rank_div for r in seed_results])
        mults = np.array([r.top_mult for r in seed_results])
        lines += [
            f"## wd={wd} (n={len(seed_results)})",
            "",
            f"- effective_rank collapse: mean {divs.mean():.2f}x, std {divs.std(ddof=1):.2f}, "
            f"range [{divs.min():.2f}, {divs.max():.2f}]x",
            f"- top_eig_corr rise: mean {mults.mean():.2f}x, std {mults.std(ddof=1):.2f}, "
            f"range [{mults.min():.2f}, {mults.max():.2f}]x",
            "",
            "| seed | onset step (2x chance) | primary onset (acc >= 0.5) | rank collapse | top_eig rise | flagged in window | onset/last flagged |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in seed_results:
            flag_note = f"{'onset' if r.onset_flagged else ''}{',' if r.onset_flagged and r.last_flagged else ''}{'last' if r.last_flagged else ''}" or "--"
            lines.append(
                f"| {r.seed} | {r.onset_step} | {_step_or_note(r.t_onset)} | {r.rank_div:.2f}x | {r.top_mult:.2f}x | "
                f"{r.n_flagged} | {flag_note} |"
            )
        lines.append("")
    lines += [
        "## Reading this",
        "",
        "This reuses the seed=0 headline definition unfiltered -- flagged (train_acc<0.9) "
        "checkpoints are not dropped from the onset/last-checkpoint lookup, only annotated in the "
        "table above. A seed whose onset or last-window checkpoint itself is flagged should be read "
        "with that in mind before trusting its individual collapse factor.",
        "",
        "The onset here is the 2x-chance crossing, a secondary definition "
        "(eigengrok/analysis/milestones.py): it is noise-dominated across seeds, and some of the seeds "
        "above cross it within the first few hundred steps, long before their primary onset (first test "
        "accuracy >= 0.5, the column next to it). A collapse factor measured from such an early step "
        "is not a post-onset collapse.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--tag", default="effective_rank_multiseed")
    args = parser.parse_args()

    results: Dict[str, List[SeedResult]] = {}
    for wd in CONDITIONS:
        seed_results = []
        for seed in SEEDS:
            parquet, run_dir = _run_paths(wd, seed)
            if not parquet.exists():
                print(f"Skipping wd={wd} seed={seed}: {parquet} not found")
                continue
            seed_results.append(_load_seed(wd, seed))
        results[wd] = seed_results

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_overlay(results)
    png_path = args.out_dir / f"{args.tag}.png"
    pdf_path = args.out_dir / f"{args.tag}.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")

    write_summary(results, args.out_dir / f"{args.tag}_summary.md")


if __name__ == "__main__":
    main()
