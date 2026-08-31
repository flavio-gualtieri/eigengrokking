#!/usr/bin/env python3
# analysis/onset_vs_wd.py

"""
Mean generalization onset vs. weight decay, one line per init scale alpha.

Answers a sweep-level question the single-run gate figure
(analysis/make_figures.py) can't: does onset move monotonically with weight
decay, and how does that compare across init scale? Regenerates entirely
from the committed accuracy CSV (scripts/build_wd_sweep_accuracy_csv.py) --
no notebook, one command:

    python -m analysis.onset_vs_wd

Onset here is defined per (init, wd, seed) run as the first logged step at
which test accuracy exceeds a fixed 0.5 threshold -- deliberately not the
2x-chance crossing analysis/transitions.py.find_onset() uses for the
single-run gate figure. That threshold (2/modulus =~ 2.2% for p=91) is fine
for one curve inspected by eye, but across this many seeds it is noise-
dominated: several runs briefly clear it within the first ~200 steps purely
from pre-training softmax noise, well before any real learning (verified by
inspection -- those runs then fall back near chance for a while before the
real transition). A fixed, far-from-chance accuracy threshold (0.5, cleared
eventually by every run in this sweep) is a cruder generalization criterion
but a much more robust one to aggregate over 10 seeds x 19 (init, wd)
settings without hand-checking every curve.

Output (see --out-dir, default reports/figures/):
    onset_vs_wd.pdf / .png    mean onset step vs. wd, one line per alpha,
                               individual seeds shown as jittered points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_ACCURACY_CSV = Path("reports/wd_sweep_accuracy_curves.csv")
ONSET_ACC_THRESHOLD = 0.5

# Okabe-Ito: colorblind-safe, print-safe (see analysis/make_figures.py). Two
# fixed-identity series (alpha=1, alpha=8) get two fixed colors, never
# reassigned by filtering/order.
COLOR_ALPHA1 = "#0072B2"
COLOR_ALPHA8 = "#D55E00"
COLORS_BY_INIT = {1: COLOR_ALPHA1, 8: COLOR_ALPHA8}


def compute_onsets(df: pd.DataFrame, *, threshold: float = ONSET_ACC_THRESHOLD) -> pd.DataFrame:
    """
    One row per (init, wd, seed): the first step test_accuracy clears
    `threshold`. Raises if any run never clears it in this sweep's logged
    range -- silently dropping a run would quietly change what the mean is
    averaged over.
    """
    def onset(g: pd.DataFrame):
        g = g.sort_values("step")
        above = g.loc[g["test_accuracy"] > threshold, "step"]
        if above.empty:
            return np.nan
        return above.iloc[0]

    out = (
        df.groupby(["init", "wd", "seed"])
        .apply(onset, include_groups=False)
        .reset_index(name="onset_step")
    )
    missing = out[out["onset_step"].isna()]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} run(s) never reach test_accuracy > {threshold} in this sweep -- "
            f"can't compute onset for them:\n{missing[['init', 'wd', 'seed']].to_string(index=False)}"
        )
    out["onset_step"] = out["onset_step"].astype(int)
    return out


def plot_onset_vs_wd(onsets: pd.DataFrame, *, threshold: float = ONSET_ACC_THRESHOLD) -> plt.Figure:
    """
    One line per init (mean onset step over seeds, log y-axis), with every
    individual seed's onset step overlaid as a small jittered point --
    spread shown raw, not summarized into a band, since a shaded band reads
    as a confidence interval this (10-seed, single-modulus) sweep can't
    support. No fitted trend line: the eye does the "monotone decrease"
    read directly off the means.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    inits = sorted(onsets["init"].unique())
    # Deterministic per-seed jitter, shared across wd/init so the same seed
    # sits at the same relative offset everywhere -- makes it easy to trace
    # one seed's onset across weight decay if a reader wants to.
    seeds = sorted(onsets["seed"].unique())
    rng = np.random.default_rng(0)
    jitter_by_seed = dict(zip(seeds, rng.uniform(-0.012, 0.012, size=len(seeds))))

    for init in inits:
        color = COLORS_BY_INIT.get(init, "#666666")
        sub = onsets[onsets["init"] == init]
        means = sub.groupby("wd")["onset_step"].mean().sort_index()

        jitter = sub["seed"].map(jitter_by_seed).to_numpy()
        ax.scatter(sub["wd"] + jitter, sub["onset_step"], color=color, s=14, alpha=0.35,
                   linewidths=0, zorder=2)
        ax.plot(means.index, means.to_numpy(), color=color, linewidth=2.2, marker="o",
                markersize=5, zorder=3, label=f"alpha={init:g} (mean of {sub['seed'].nunique()} seeds)")

    ax.set_yscale("log")
    ax.set_xlabel("Weight decay")
    ax.set_ylabel(f"Onset step (first test_accuracy > {threshold:g})")
    ax.set_title("Generalization onset vs. weight decay")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.2)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accuracy-csv", type=Path, default=DEFAULT_ACCURACY_CSV)
    parser.add_argument("--threshold", type=float, default=ONSET_ACC_THRESHOLD,
                         help="Test-accuracy level defining onset (default: 0.5)")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--tag", default="onset_vs_wd", help="Output filename stem")
    args = parser.parse_args()

    df = pd.read_csv(args.accuracy_csv)
    onsets = compute_onsets(df, threshold=args.threshold)

    summary = onsets.groupby(["init", "wd"])["onset_step"].agg(["mean", "std", "count"])
    print(summary.to_string())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_onset_vs_wd(onsets, threshold=args.threshold)

    png_path = args.out_dir / f"{args.tag}.png"
    fig.savefig(png_path, dpi=200)
    print(f"Wrote {png_path}")

    pdf_path = args.out_dir / f"{args.tag}.pdf"
    fig.savefig(pdf_path)
    print(f"Wrote {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
