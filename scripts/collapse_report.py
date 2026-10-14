#!/usr/bin/env python3
# scripts/collapse_report.py

"""
Report on the Phase 1 collapse pilot (configs/collapse_pilot.yaml,
scripts/run_collapse_pilot_array.sh): post-grokking collapses, milestones and
the instability diagnostics per arm, against arm A (the sweep's settings) as
the baseline, and arm A against the sweep's own wd=1.0, init=1 runs.

    python scripts/collapse_report.py               # extract runs/collapse_pilot/, then report
    python scripts/collapse_report.py --from-csv    # report from the committed curves only

Arms with no finished runs yet are listed as missing, so the report can be
rerun as the array job progresses. Milestones and collapses:
eigengrok/analysis/milestones.py. Intervals: 95% percentile bootstrap over seeds.

Writes:
    reports/collapse_pilot_curves.csv           every logged point of every run (the pilot's data)
    reports/collapse_pilot_runs.csv             one row per run
    reports/collapse_pilot_summary.md           per-arm tables and comparisons
    reports/figures/collapse_pilot_test_acc.{png,pdf}   test accuracy, all seeds, one panel per arm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eigengrok.analysis.bootstrap import bootstrap_ci, bootstrap_ratio_ci
from eigengrok.analysis.milestones import (
    COLLAPSE_HIGH,
    COLLAPSE_LOW,
    count_collapses,
    t_grok,
    t_mem,
    t_onset,
)

ARMS = {  # name -> what differs from the sweep's settings
    "A_default": "sweep settings: AdamW betas (0.9, 0.999), no clipping, float32",
    "B_beta2_0.98": "betas (0.9, 0.98)",
    "C_beta2_0.98_clip1": "betas (0.9, 0.98), gradient clipping at 1.0",
    "D_float64": "sweep settings in float64",
}
BASELINE = "A_default"
CURVE_KEYS = {  # npz key -> CSV column
    "train_accuracies": "train_accuracy", "test_accuracies": "test_accuracy",
    "train_losses": "train_loss", "test_losses": "test_loss",
    "norms": "norm", "last_layer_norms": "last_layer_norm", "grad_norms": "grad_norm",
    "update_norms": "update_norm", "max_abs_logits": "max_abs_logit", "frac_saturated": "frac_saturated",
}
DIAGNOSTICS = ("grad_norm", "update_norm", "max_abs_logit", "frac_saturated", "last_layer_norm")
SWEEP_CSV = Path("reports/wd_sweep_accuracy_curves.csv")
COLOR = "#0072B2"
COLOR_EVENT = "#D55E00"


def extract(pilot_dir: Path) -> pd.DataFrame:
    frames = []
    for npz_path in sorted(pilot_dir.glob("*/**/figures/training_data_*.npz")):
        run_dir = npz_path.parent.parent
        arm = npz_path.relative_to(pilot_dir).parts[0]
        cfg = json.loads((run_dir / "config_full.json").read_text())
        with np.load(npz_path) as z:
            if not np.array_equal(z["train_steps"], z["test_steps"]):
                raise ValueError(f"{npz_path}: train/test step grids differ")
            frame = pd.DataFrame({"step": z["train_steps"], **{col: z[key] for key, col in CURVE_KEYS.items()}})
        frames.append(frame.assign(arm=arm, seed=int(cfg["seed"]), adam_beta2=cfg["adam_beta2"],
                                   grad_clip=cfg["grad_clip"], dtype=cfg["dtype"]))
    if not frames:
        raise FileNotFoundError(f"No finished runs under {pilot_dir}")
    cols = ["arm", "seed", "adam_beta2", "grad_clip", "dtype", "step", *CURVE_KEYS.values()]
    return pd.concat(frames)[cols].sort_values(["arm", "seed", "step"]).reset_index(drop=True)


def collapse_events(test_acc: np.ndarray) -> list[int]:
    """Index of the first logged point of each collapse (same rule as milestones.count_collapses)."""
    armed, events = False, []
    for i, v in enumerate(test_acc):
        if v >= COLLAPSE_HIGH:
            armed = True
        elif v < COLLAPSE_LOW and armed:
            events.append(i)
            armed = False
    return events


def run_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, seed), g in curves.groupby(["arm", "seed"], sort=True):
        g = g.sort_values("step")
        steps, test = g["step"].to_numpy(), g["test_accuracy"].to_numpy()
        row = {"arm": arm, "seed": seed, "t_mem": t_mem(steps, g["train_accuracy"]), "t_onset": t_onset(steps, test),
               "t_grok": t_grok(steps, test), "n_collapses": count_collapses(test),
               "n_train_collapses": count_collapses(g["train_accuracy"]), "final_test_acc": test[-1],
               "max_frac_saturated": g["frac_saturated"].max(), "max_abs_logit": g["max_abs_logit"].max()}
        # Diagnostics at the last log point before each collapse and at its first point below
        # 0.5, relative to the run's median after grokking (the regime the collapses happen in).
        post = g[g["step"] >= row["t_grok"]] if np.isfinite(row["t_grok"]) else g.iloc[0:0]
        events = collapse_events(test)
        for d in DIAGNOSTICS:
            ref = post[d].median() if len(post) else np.nan
            for when, offset in (("before", -1), ("at", 0)):
                vals = [g[d].iloc[i + offset] for i in events if i + offset >= 0]
                row[f"{d}_{when}_collapse_rel"] = float(np.median(vals) / ref) if vals and ref else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_baseline(max_step: int) -> pd.DataFrame:
    """The sweep's wd=1.0, init=1 runs over the pilot's step range, for the arm A reproduction check."""
    df = pd.read_csv(SWEEP_CSV)
    df = df[(df["init"] == 1) & np.isclose(df["wd"], 1.0) & (df["step"] <= max_step)]
    rows = []
    for seed, g in df.groupby("seed"):
        g = g.sort_values("step")
        rows.append({"seed": seed, "t_onset": t_onset(g["step"], g["test_accuracy"]),
                     "t_grok": t_grok(g["step"], g["test_accuracy"]),
                     "n_collapses": count_collapses(g["test_accuracy"].to_numpy())})
    return pd.DataFrame(rows)


def _ci(lo_hi, fmt="{:.1f}") -> str:
    lo, hi = lo_hi
    return "--" if not np.isfinite(lo) else f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def write_markdown(path: Path, runs: pd.DataFrame, sweep: pd.DataFrame, max_step: int, source: str) -> None:
    present = [a for a in ARMS if a in set(runs["arm"])]
    missing = [a for a in ARMS if a not in present]
    L = [
        "# Collapse pilot (Phase 1)",
        "",
        f"Generated by `python scripts/collapse_report.py` from {source}. p=91, wd=1.0, init=1, "
        f"logged every 100 steps to step {max_step:,}. Collapse: test accuracy falling from >= {COLLAPSE_HIGH} "
        f"to < {COLLAPSE_LOW} (`eigengrok/analysis/milestones.py`). Intervals: 95% bootstrap over seeds.",
        "",
        "Arms: " + "; ".join(f"`{a}` = {d}" for a, d in ARMS.items()) + ".",
        "",
    ]
    if missing:
        L += [f"**Incomplete:** no finished runs yet for {', '.join(f'`{a}`' for a in missing)}.", ""]
    L += [
        "## Per arm",
        "",
        "| arm | seeds | collapses / run: mean [CI] | median (range) | train collapses / run | "
        "t_onset median [CI] | t_grok median [CI] | final test acc >= 0.95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in present:
        r = runs[runs["arm"] == arm]
        c = r["n_collapses"].to_numpy(float)
        L.append(
            f"| `{arm}` | {len(r)} | {c.mean():.1f} {_ci(bootstrap_ci(c, np.mean))} | "
            f"{np.median(c):.0f} ({c.min():.0f}-{c.max():.0f}) | {r['n_train_collapses'].mean():.1f} | "
            f"{r['t_onset'].median():,.0f} {_ci(bootstrap_ci(r['t_onset'], np.median), '{:,.0f}')} | "
            f"{r['t_grok'].median():,.0f} {_ci(bootstrap_ci(r['t_grok'], np.median), '{:,.0f}')} | "
            f"{(r['final_test_acc'] >= 0.95).mean():.0%} |"
        )
    base = runs[runs["arm"] == BASELINE]
    others = [a for a in present if a != BASELINE]
    if len(base) and others:
        L += ["", f"## Against `{BASELINE}` (baseline)", "",
              "| arm | ratio of mean collapses / run [CI] | ratio of median t_onset [CI] |", "|---|---|---|"]
        for arm in others:
            r = runs[runs["arm"] == arm]
            L.append(
                f"| `{arm}` | {r['n_collapses'].mean() / base['n_collapses'].mean():.2f} "
                f"{_ci(bootstrap_ratio_ci(r['n_collapses'], base['n_collapses'], np.mean), '{:.2f}')} | "
                f"{r['t_onset'].median() / base['t_onset'].median():.2f} "
                f"{_ci(bootstrap_ratio_ci(r['t_onset'], base['t_onset'], np.median), '{:.2f}')} |"
            )
    L += [
        "",
        "## Diagnostics",
        "",
        "Run maxima of the softmax-saturation fraction (training examples with p_max > 1 - 1e-7) and of max |logit|; "
        "then each diagnostic at the last log point before a collapse and at its first point below 0.5, divided by "
        "the run's median after `t_grok` (median over the run's collapses, then median (range) over runs; 1.00x = "
        "typical post-grokking value). Log points are 100 steps apart, so anything faster than that is invisible here.",
        "",
        "| arm | max saturated fraction | max abs logit | diagnostic | before collapse | at collapse |",
        "|---|---|---|---|---|---|",
    ]
    for arm in present:
        r = runs[runs["arm"] == arm]
        for k, d in enumerate(DIAGNOSTICS):
            cells = []
            for when in ("before", "at"):
                v = r[f"{d}_{when}_collapse_rel"].dropna()
                cells.append("--" if v.empty else f"{v.median():.2f}x ({v.min():.2f}-{v.max():.2f})")
            head = (f"`{arm}` | {r['max_frac_saturated'].max():.3g} | {r['max_abs_logit'].max():.3g}"
                    if k == 0 else " | | ")
            L.append(f"| {head} | {d} | {cells[0]} | {cells[1]} |")
    if len(base):
        both = base.merge(sweep, on="seed", suffixes=("_pilot", "_sweep"))
        same_onset = int((both["t_onset_pilot"] == both["t_onset_sweep"]).sum())
        L += [
            "",
            f"## `{BASELINE}` against the sweep's wd=1.0, init=1 runs (same seeds, steps <= {max_step:,})",
            "",
            "Same code path and settings (bitwise identical on CPU); on GPU the two can drift apart through "
            "non-deterministic kernels.",
            "",
            "| | collapses / run: mean [CI] | t_onset median [CI] |",
            "|---|---|---|",
            f"| pilot `{BASELINE}` | {both['n_collapses_pilot'].mean():.1f} "
            f"{_ci(bootstrap_ci(both['n_collapses_pilot'], np.mean))} | {both['t_onset_pilot'].median():,.0f} "
            f"{_ci(bootstrap_ci(both['t_onset_pilot'], np.median), '{:,.0f}')} |",
            f"| sweep | {both['n_collapses_sweep'].mean():.1f} {_ci(bootstrap_ci(both['n_collapses_sweep'], np.mean))} | "
            f"{both['t_onset_sweep'].median():,.0f} {_ci(bootstrap_ci(both['t_onset_sweep'], np.median), '{:,.0f}')} |",
            "",
            f"Seeds with the same `t_onset` in both: {same_onset} of {len(both)}.",
        ]
    L.append("")
    path.write_text("\n".join(L))


def plot(curves: pd.DataFrame, path_stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (arm, desc) in zip(axes.flat, ARMS.items()):
        sub = curves[curves["arm"] == arm]
        for _, g in sub.groupby("seed"):
            ax.plot(g["step"], g["test_accuracy"], color=COLOR, linewidth=0.8, alpha=0.55)
            idx = collapse_events(g["test_accuracy"].to_numpy())
            ax.scatter(g["step"].iloc[idx], g["test_accuracy"].iloc[idx], color=COLOR_EVENT, s=10, zorder=3, linewidths=0)
        n = sub["seed"].nunique()
        ax.set_title(f"{arm} ({n} seeds)" if n else f"{arm} (no runs yet)", fontsize=10)
        ax.text(0.98, 0.04, desc, transform=ax.transAxes, ha="right", fontsize=7.5, color="#555555")
        ax.axhline(COLLAPSE_LOW, color="#999999", linewidth=0.6, linestyle="--")
        ax.grid(True, alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Optimization step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test accuracy")
    handles = [plt.Line2D([], [], color=COLOR, linewidth=0.8, label="one seed"),
               plt.Line2D([], [], color=COLOR_EVENT, marker="o", linestyle="", markersize=4,
                          label="collapse (first point < 0.5)")]
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Collapse pilot: test accuracy per seed, p=91, wd=1.0, init=1", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot-dir", type=Path, default=Path("runs/collapse_pilot"))
    parser.add_argument("--from-csv", action="store_true", help="Use reports/collapse_pilot_curves.csv, not the run directories")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    curves_csv = args.out_dir / "collapse_pilot_curves.csv"
    if args.from_csv:
        curves, source = pd.read_csv(curves_csv), f"`{curves_csv}`"
    else:
        curves, source = extract(args.pilot_dir), f"`{args.pilot_dir}/` (copied to `{curves_csv}`)"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        curves.to_csv(curves_csv, index=False, float_format="%.9g")
        print(f"Wrote {curves_csv}")

    runs = run_table(curves)
    max_step = int(curves["step"].max())
    runs.to_csv(args.out_dir / "collapse_pilot_runs.csv", index=False, float_format="%.6g")
    write_markdown(args.out_dir / "collapse_pilot_summary.md", runs, sweep_baseline(max_step), max_step, source)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    plot(curves, args.out_dir / "figures" / "collapse_pilot_test_acc")
    for name in ("collapse_pilot_runs.csv", "collapse_pilot_summary.md", "figures/collapse_pilot_test_acc.png"):
        print(f"Wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
