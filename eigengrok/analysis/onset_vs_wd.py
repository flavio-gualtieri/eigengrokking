#!/usr/bin/env python3
# eigengrok/analysis/onset_vs_wd.py

"""
Mean generalization onset vs. weight decay, one line per init scale alpha.

Answers a sweep-level question the single-run gate figure
(eigengrok/analysis/make_figures.py) can't: does onset move monotonically with weight
decay, and how does that compare across init scale? Regenerates entirely
from the committed accuracy CSV (scripts/build_wd_sweep_accuracy_csv.py) --
no notebook, one command:

    python -m eigengrok.analysis.onset_vs_wd

Onset is the primary definition, t_onset in eigengrok/analysis/milestones.py:
the first logged step with test accuracy >= 0.5. Not the 2x-chance crossing
the single-run gate figure aligns on (eigengrok/analysis/transitions.py):
across this many seeds that one is noise-dominated, with several runs
clearing it within the first few hundred steps, before memorization, then
falling back to chance until the real transition. 0.5 is cleared eventually
by every run in the sweep (12 weight decays x 2 init scales x 10 seeds).

Output (see --out-dir, default reports/figures/):
    onset_vs_wd.pdf / .png    mean onset step vs. wd, one line per alpha,
                               with a 95% bootstrap CI over seeds; individual
                               seeds shown as jittered points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eigengrok.analysis.bootstrap import bootstrap_ci
from eigengrok.analysis.milestones import ONSET_TEST_ACC, RUN_KEYS, first_step_at_or_above

DEFAULT_ACCURACY_CSV = Path("reports/wd_sweep_accuracy_curves.csv")

# Okabe-Ito: colorblind-safe, print-safe (see eigengrok/analysis/make_figures.py). Two
# fixed-identity series (alpha=1, alpha=8) get two fixed colors, never
# reassigned by filtering/order.
COLOR_ALPHA1 = "#0072B2"
COLOR_ALPHA8 = "#D55E00"
COLORS_BY_INIT = {1: COLOR_ALPHA1, 8: COLOR_ALPHA8}


def compute_onsets(df: pd.DataFrame, *, threshold: float = ONSET_TEST_ACC) -> pd.DataFrame:
    """
    One row per (init, wd, seed): the first step test_accuracy reaches
    `threshold` (default: milestones.t_onset's 0.5). Raises if any run never
    reaches it in this sweep's logged range -- silently dropping a run would
    quietly change what the mean is averaged over.
    """
    out = (
        df.groupby(list(RUN_KEYS))
        .apply(lambda g: first_step_at_or_above(g["step"], g["test_accuracy"], threshold), include_groups=False)
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


def plot_onset_vs_wd(onsets: pd.DataFrame, *, threshold: float = ONSET_TEST_ACC) -> plt.Figure:
    """
    One line per init (mean onset step over seeds, log y-axis) with a 95%
    percentile-bootstrap CI of that mean over seeds (eigengrok/analysis/bootstrap.py),
    and every individual seed's onset step overlaid as a small jittered
    point, so the raw spread stays visible next to the interval. The two
    lines are dodged by +-0.005 in wd so their error bars don't overlap.
    No fitted trend line.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    inits = sorted(onsets["init"].unique())
    # Deterministic per-seed jitter, shared across wd/init so the same seed
    # sits at the same relative offset everywhere -- makes it easy to trace
    # one seed's onset across weight decay if a reader wants to.
    seeds = sorted(onsets["seed"].unique())
    rng = np.random.default_rng(0)
    jitter_by_seed = dict(zip(seeds, rng.uniform(-0.012, 0.012, size=len(seeds))))

    dodge = dict(zip(inits, np.linspace(-0.005, 0.005, len(inits)) if len(inits) > 1 else [0.0]))
    for init in inits:
        color = COLORS_BY_INIT.get(init, "#666666")
        sub = onsets[onsets["init"] == init]
        by_wd = sub.groupby("wd")["onset_step"]
        means = by_wd.mean().sort_index()
        ci = np.array([bootstrap_ci(by_wd.get_group(wd), np.mean) for wd in means.index])
        yerr = np.vstack([means.to_numpy() - ci[:, 0], ci[:, 1] - means.to_numpy()])

        jitter = sub["seed"].map(jitter_by_seed).to_numpy()
        ax.scatter(sub["wd"] + jitter, sub["onset_step"], color=color, s=14, alpha=0.35,
                   linewidths=0, zorder=2)
        ax.errorbar(means.index + dodge[init], means.to_numpy(), yerr=yerr, color=color, linewidth=2.2,
                    marker="o", markersize=5, capsize=3, elinewidth=1.2, zorder=3,
                    label=f"alpha={init:g} (mean of {sub['seed'].nunique()} seeds, 95% bootstrap CI)")

    ax.set_yscale("log")
    ax.set_xlabel("Weight decay")
    ax.set_ylabel(f"Onset step (first test accuracy >= {threshold:g})")
    ax.set_title("Generalization onset vs. weight decay")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.2)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accuracy-csv", type=Path, default=DEFAULT_ACCURACY_CSV)
    parser.add_argument("--threshold", type=float, default=ONSET_TEST_ACC,
                         help="Test-accuracy level defining onset (default: milestones.t_onset's 0.5)")
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
