#!/usr/bin/env python3
# scripts/figures/forecast_figures.py

"""
Phase 3 figures, from the outputs of scripts/forecast_benchmark.py and the
sweep's spectra (scripts/run_spectral_sweep_array.sh):

    python scripts/figures/forecast_figures.py

    reports/figures/forecast_increments.{png,pdf}   (i) gain in R^2 and R^2 within settings from each
                                                    indicator family, per split, with 95% CIs
    reports/figures/forecast_pred_vs_actual.{png,pdf}  (ii) predicted vs actual ln(t_onset), coloured by wd
    reports/figures/forecast_trajectory.{png,pdf}   (iii) one run's curvature aligned to its onset, with the
                                                    median and interquartile band over its setting's seeds

Figures needing curvature are skipped (with a message) until the spectra exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eigengrok.analysis.forecast_features import load_spectra

# Okabe-Ito, fixed order; each series also has its own marker (secondary encoding for colour-vision deficiency).
INCREMENTS = [  # (label, feature set whose delta vs B0+B1 is used, sign, colour, marker)
    ("cheap statistics (B1) over hyperparameters (B0)", "B0", -1, "#0072B2", "o"),
    ("curvature H + G over B0+B1", "B0+B1+C", 1, "#D55E00", "s"),
    ("curvature H only over B0+B1", "B0+B1+C(H only)", 1, "#009E73", "^"),
    ("curvature G only over B0+B1", "B0+B1+C(G only)", 1, "#CC79A7", "D"),
]
# The init-scale split is left out: trained on one init scale, no model transfers to the other (every
# R^2 is far below zero; reports/forecast_summary.md has the numbers), so its intervals would swamp the axes.
SPLITS = ("settings", "wd_range")
SPLIT_LABELS = {"settings": "leave one\nsetting out", "init": "leave one\ninit scale out", "wd_range": "train wd <= 0.7,\ntest wd > 0.7"}
INK = "#333333"


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {stem}.png/.pdf")


def increments(results: pd.DataFrame, cutoff: str, model: str, stem: Path) -> None:
    r = results[(results["cutoff"] == cutoff) & (results["model"] == model)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, metric, title in zip(axes, ("r2", "r2_within"), ("R^2 of ln(t_onset)", "R^2 within settings (seed level)")):
        x = np.arange(len(SPLITS))
        shown = [inc for inc in INCREMENTS if (r["features"] == inc[1]).any()]
        width = 0.18
        for i, (label, fs, sign, color, marker) in enumerate(shown):
            ys, lo, hi = [], [], []
            for split in SPLITS:
                row = r[(r["split"] == split) & (r["features"] == fs) & (r["metric"] == metric)]
                d = row["delta"].iloc[0] if len(row) and "delta" in row else np.nan
                a, b = (row["delta_ci_lo"].iloc[0], row["delta_ci_hi"].iloc[0]) if len(row) else (np.nan, np.nan)
                d, a, b = sign * d, *(sorted([sign * a, sign * b]))
                ys.append(d); lo.append(d - a); hi.append(b - d)
            pos = x + (i - (len(shown) - 1) / 2) * width
            ax.errorbar(pos, ys, yerr=[lo, hi], fmt=marker, color=color, markersize=6, capsize=3, linewidth=1.2,
                        label=label)
        ax.axhline(0, color=INK, linewidth=0.8)
        lo_lim, hi_lim = ax.get_ylim()
        clipped = lo_lim < -1 or hi_lim > 1
        ax.set_ylim(max(lo_lim, -1.05), min(hi_lim, 1.05))  # an R^2 gain beyond +-1 only means "too few runs to tell"
        ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLITS], fontsize=8.5)
        ax.set_title(title + (" (axis clipped at +-1)" if clipped else ""), fontsize=10)
        ax.set_ylabel("gain (95% CI)")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="best", frameon=False)
    fig.suptitle(f"What each indicator family adds, ridge, cutoff {cutoff} (init-scale split omitted: no model "
                 "transfers across init scales)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, stem)


def pred_vs_actual(preds: pd.DataFrame, stem: Path) -> None:
    sets = [c for c in ("pred:B0+B1", "pred:B0+B1+C") if c in preds]
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 4.6), squeeze=False)
    cmap = plt.get_cmap("Blues")
    wds = np.sort(preds["wd"].unique())
    norm = plt.Normalize(wds.min() - 0.25 * np.ptp(wds), wds.max())  # keep the lightest wd visible
    lim = [min(preds["y"].min(), np.nanmin(preds[sets].to_numpy())) - 0.1,
           max(preds["y"].max(), np.nanmax(preds[sets].to_numpy())) + 0.1]
    for ax, col in zip(axes[0], sets):
        for init, marker in ((1.0, "o"), (8.0, "^")):
            sub = preds[preds["init"] == init]
            sc = ax.scatter(sub["y"], sub[col], c=sub["wd"], cmap=cmap, norm=norm, marker=marker, s=22,
                            edgecolors=INK, linewidths=0.3, label=f"init {init:g}")
        ax.plot(lim, lim, color=INK, linewidth=0.8, linestyle="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("actual ln(t_onset)")
        if ax is axes[0][0]:
            ax.set_ylabel("predicted (out of fold)")
        ax.set_title(col.removeprefix("pred:"), fontsize=10)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax.grid(True, alpha=0.2)
    fig.colorbar(sc, ax=axes[0].tolist(), label="weight decay")
    fig.suptitle("Predicted vs actual onset, ridge, leave one setting out", fontsize=11, x=0.01, ha="left")
    _save(fig, stem)


def trajectory(spectra: pd.DataFrame, preds: pd.DataFrame, stem: Path, *, init: float = 1.0, wd: float = 1.0) -> None:
    runs = preds[(preds["init"] == init) & np.isclose(preds["wd"], wd)]
    sp = spectra[(spectra["init"] == init) & np.isclose(spectra["wd"], wd)]
    if runs.empty or sp.empty:
        print(f"Skipping the trajectory figure: no spectra for init={init:g}, wd={wd:g} yet")
        return
    rep = runs.iloc[(runs["t_onset"] - runs["t_onset"].median()).abs().argmin()]  # the median-onset seed
    panels = [("top_eig", "H top eigenvalue"), ("effective_rank", "H effective rank"),
              ("gn_top_eig", "G top eigenvalue"), ("gn_effective_rank", "G effective rank")]
    grid = np.logspace(np.log10(0.05), np.log10(3), 60)  # step / t_onset
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        curves = []
        for _, run in runs.iterrows():
            g = sp[sp["seed"] == run["seed"]].sort_values("step")
            if len(g) > 1:
                curves.append(np.interp(np.log(grid), np.log(g["step"] / run["t_onset"]), np.log(g[col]),
                                        left=np.nan, right=np.nan))
        curves = np.array(curves)
        q = np.nanquantile(curves, [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(grid, np.exp(q[0]), np.exp(q[2]), color="#0072B2", alpha=0.2, linewidth=0,
                        label=f"interquartile band, {len(curves)} seeds")
        ax.plot(grid, np.exp(q[1]), color="#0072B2", linewidth=1.2, label="median over seeds")
        g = sp[sp["seed"] == rep["seed"]].sort_values("step")
        ax.errorbar(g["step"] / rep["t_onset"], g[col], yerr=g.get(f"{col}_stderr"), color="#D55E00", linewidth=1,
                    marker="o", markersize=2.5, capsize=1.5, errorevery=max(1, len(g) // 25),
                    label=f"seed {int(rep['seed'])} (stderr over probes)")
        ax.axvline(1, color=INK, linewidth=0.8, linestyle="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("step / t_onset")
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"Curvature around onset, init={init:g}, wd={wd:g}", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--spectra-dir", type=Path, default=Path("reports/spectra_sweep"))
    parser.add_argument("--cutoff", default="t_mem+500")
    parser.add_argument("--model", default="ridge")
    args = parser.parse_args()

    out = args.reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.reports / "forecast_results.csv")
    preds = pd.read_csv(args.reports / "forecast_predictions.csv")
    increments(results, args.cutoff, args.model, out / "forecast_increments")
    pred_vs_actual(preds, out / "forecast_pred_vs_actual")
    spectra = load_spectra(args.spectra_dir) if args.spectra_dir.exists() else pd.DataFrame()
    if spectra.empty:
        print("Skipping the trajectory figure: no spectra yet")
    else:
        trajectory(spectra, preds, out / "forecast_trajectory")


if __name__ == "__main__":
    main()
