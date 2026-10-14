#!/usr/bin/env python3
# scripts/sweep_summary.py

"""
Milestone summary of the p=91 weight-decay x init-scale sweep, computed from
the committed accuracy curves alone (reports/wd_sweep_accuracy_curves.csv,
written by scripts/build_wd_sweep_accuracy_csv.py):

    python scripts/sweep_summary.py

Milestone definitions: eigengrok/analysis/milestones.py. Intervals: 95%
percentile bootstrap over seeds (eigengrok/analysis/bootstrap.py, 10,000
resamples, seed 0); statistics pooled over weight decays resample seeds
within each weight decay.

Writes (see --out-dir, default reports/):
    sweep_milestones.csv  one row per run: every milestone, final test accuracy, collapse counts
    sweep_summary.csv     one row per (init, wd): mean and median of each, with 95% CIs
    sweep_summary.md      the per-(init, wd) tables plus the init-scale, weight-decay,
                          collapse and onset-definition comparisons, each against a stated baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eigengrok.analysis.bootstrap import LEVEL, N_BOOT, bootstrap_ci, bootstrap_ratio_ci
from eigengrok.analysis.milestones import (
    COLLAPSE_HIGH,
    COLLAPSE_LOW,
    GROK_SUSTAIN_EVALS,
    GROK_TEST_ACC,
    MEM_TRAIN_ACC,
    ONSET_TEST_ACC,
    milestones_table,
)

DEFAULT_ACCURACY_CSV = Path("reports/wd_sweep_accuracy_curves.csv")
STEP_METRICS = ("t_mem", "t_onset", "t_grok", "gap", "onset_2x_chance", "onset_logistic")
INIT_BASELINE = 1      # init-scale comparisons: init 8 against init 1 at the same weight decay
WD_BASELINE = 1.0      # weight-decay comparisons: each wd against wd = 1.0 at the same init
WD_LOW = 0.45
FINAL_ACC_OK = 0.95


def gmean(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.exp(np.mean(np.log(x), axis=axis))


# --- per (init, wd) ---

def cell_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (init, wd), g in runs.groupby(["init", "wd"], sort=True):
        row: dict = {"init": init, "wd": wd, "n_runs": len(g)}
        for m in STEP_METRICS:
            x = g[m].to_numpy(dtype=float)
            finite = x[np.isfinite(x)]
            row[f"{m}_n"] = len(finite)
            for name, stat in (("mean", np.mean), ("median", np.median)):
                row[f"{m}_{name}"] = float(stat(finite)) if len(finite) else float("nan")
                row[f"{m}_{name}_ci_lo"], row[f"{m}_{name}_ci_hi"] = bootstrap_ci(x, stat)
        row["n_2x_chance_before_mem"] = int((g["onset_2x_chance"] < g["t_mem"]).sum())

        acc = g["final_test_acc"].to_numpy(dtype=float)
        row["final_test_acc_mean"] = float(acc.mean())
        row["final_test_acc_mean_ci_lo"], row["final_test_acc_mean_ci_hi"] = bootstrap_ci(acc, np.mean)
        row["final_test_acc_min"] = float(acc.min())
        row[f"final_test_acc_frac_ge_{FINAL_ACC_OK}"] = float((acc >= FINAL_ACC_OK).mean())

        for col in ("n_collapses", "n_train_collapses"):
            c = g[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(c.mean())
            row[f"{col}_mean_ci_lo"], row[f"{col}_mean_ci_hi"] = bootstrap_ci(c, np.mean)
            row[f"{col}_median"] = float(np.median(c))
            row[f"{col}_min"] = int(c.min())
            row[f"{col}_max"] = int(c.max())
        rows.append(row)
    return pd.DataFrame(rows)


# --- comparisons ---

def init_effect(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """t_onset of init 8 relative to init 1 at each weight decay, and pooled over weight decays."""
    other = sorted(set(runs["init"]) - {INIT_BASELINE})
    if len(other) != 1:
        raise ValueError(f"Expected exactly one init besides {INIT_BASELINE}, got {other}")
    alt = other[0]
    base, test = runs[runs["init"] == INIT_BASELINE], runs[runs["init"] == alt]

    rows = []
    for wd in sorted(runs["wd"].unique()):
        b = base.loc[base["wd"] == wd, "t_onset"].to_numpy(dtype=float)
        t = test.loc[test["wd"] == wd, "t_onset"].to_numpy(dtype=float)
        row = {"wd": wd}
        for name, stat in (("mean", np.mean), ("median", np.median)):
            row[f"ratio_{name}"] = float(stat(t) / stat(b))
            row[f"ratio_{name}_ci"] = bootstrap_ratio_ci(t, b, stat)
        rows.append(row)
    per_wd = pd.DataFrame(rows)

    pooled = {"alt": alt}
    for metric in ("t_onset", "t_mem", "gap"):
        for init, sub in ((INIT_BASELINE, base), (alt, test)):
            x = sub[metric].to_numpy(dtype=float)
            pooled[f"{metric}_median_init{init}"] = float(np.median(x))
            pooled[f"{metric}_median_init{init}_ci"] = bootstrap_ci(x, np.median, strata=sub["wd"])
            pooled[f"{metric}_mean_init{init}"] = float(np.mean(x))
            pooled[f"{metric}_mean_init{init}_ci"] = bootstrap_ci(x, np.mean, strata=sub["wd"])
    t, b = test["t_onset"].to_numpy(dtype=float), base["t_onset"].to_numpy(dtype=float)
    pooled["onset_gmean_ratio"] = float(gmean(t) / gmean(b))
    pooled["onset_gmean_ratio_ci"] = bootstrap_ratio_ci(t, b, gmean, strata_num=test["wd"], strata_den=base["wd"])
    tm, bm = test["t_mem"].to_numpy(dtype=float), base["t_mem"].to_numpy(dtype=float)
    pooled["mem_median_ratio"] = float(np.median(tm) / np.median(bm))
    pooled["mem_median_ratio_ci"] = bootstrap_ratio_ci(tm, bm, np.median, strata_num=test["wd"], strata_den=base["wd"])
    tg, bg = test["gap"].to_numpy(dtype=float), base["gap"].to_numpy(dtype=float)
    pooled["gap_median_ratio"] = float(np.median(tg) / np.median(bg))
    pooled["gap_median_ratio_ci"] = bootstrap_ratio_ci(tg, bg, np.median, strata_num=test["wd"], strata_den=base["wd"])
    pooled["n_negative_gap"] = int((runs["gap"] < 0).sum())
    return per_wd, pooled


def wd_effect(runs: pd.DataFrame) -> pd.DataFrame:
    """t_onset at wd = 0.45 relative to wd = 1.0, per init."""
    rows = []
    for init, g in runs.groupby("init", sort=True):
        lo = g.loc[np.isclose(g["wd"], WD_LOW), "t_onset"].to_numpy(dtype=float)
        hi = g.loc[np.isclose(g["wd"], WD_BASELINE), "t_onset"].to_numpy(dtype=float)
        row = {"init": init, "n_low": len(lo), "n_base": len(hi)}
        for name, stat in (("mean", np.mean), ("median", np.median)):
            row[f"ratio_{name}"] = float(stat(lo) / stat(hi))
            row[f"ratio_{name}_ci"] = bootstrap_ratio_ci(lo, hi, stat)
        rho, p = spearmanr(g["wd"], g["t_onset"])
        row["spearman_wd_onset"], row["spearman_p"] = float(rho), float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def collapse_stats(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in [("all", runs)] + [(f"init {i}", g) for i, g in runs.groupby("init", sort=True)]:
        for col in ("n_collapses", "n_train_collapses"):
            c = g[col].to_numpy(dtype=float)
            q = np.quantile(c, [0, 0.25, 0.5, 0.75, 1])
            rows.append({"runs": label, "curve": "test" if col == "n_collapses" else "train", "n": len(c),
                         "frac_ge_1": float((c >= 1).mean()), "min": q[0], "q25": q[1], "median": q[2],
                         "q75": q[3], "max": q[4], "mean": float(c.mean()),
                         "mean_ci": bootstrap_ci(c, np.mean, strata=g["init"].astype(str) + "/" + g["wd"].astype(str))})
    return pd.DataFrame(rows)


def onset_definition_stats(runs: pd.DataFrame) -> dict:
    out = {"n_runs": len(runs)}
    for sec in ("onset_2x_chance", "onset_logistic"):
        ok = runs[[sec, "t_onset"]].dropna()
        out[f"{sec}_n"] = len(ok)
        out[f"{sec}_spearman"] = float(spearmanr(ok[sec], ok["t_onset"])[0])
        ratio = ok[sec] / ok["t_onset"]
        out[f"{sec}_ratio_median"] = float(ratio.median())
        out[f"{sec}_ratio_q"] = (float(ratio.quantile(0.1)), float(ratio.quantile(0.9)))
    out["n_2x_before_mem"] = int((runs["onset_2x_chance"] < runs["t_mem"]).sum())
    out["n_2x_le_300"] = int((runs["onset_2x_chance"] <= 300).sum())
    both = runs[["onset_2x_chance", "onset_logistic"]].dropna()
    r = both["onset_logistic"] / both["onset_2x_chance"]
    out["n_disagree_1p2"] = int(((r > 1.2) | (r < 1 / 1.2)).sum())
    out["n_both"] = len(both)
    return out


# --- markdown ---

def _step(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:,.0f}"


def _ci(ci: tuple[float, float], fmt=_step) -> str:
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def _r(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:.2f}"


def _med(row, m: str) -> str:
    return f"{_step(row[f'{m}_median'])} {_ci((row[f'{m}_median_ci_lo'], row[f'{m}_median_ci_hi']))}"


def _mean(row, m: str) -> str:
    return f"{_step(row[f'{m}_mean'])} {_ci((row[f'{m}_mean_ci_lo'], row[f'{m}_mean_ci_hi']))}"


def write_markdown(path: Path, *, csv_path: Path, runs: pd.DataFrame, cells: pd.DataFrame,
                   init_wd: pd.DataFrame, init_pooled: dict, wd_eff: pd.DataFrame,
                   collapses: pd.DataFrame, onsets: dict) -> None:
    alt = init_pooled["alt"]
    n_inits, n_wds, n_seeds = runs["init"].nunique(), runs["wd"].nunique(), runs.groupby(["init", "wd"]).size()
    seeds = f"{n_seeds.min()}" if n_seeds.min() == n_seeds.max() else f"{n_seeds.min()}-{n_seeds.max()}"
    L = [
        "# Sweep summary: grokking milestones, p=91",
        "",
        f"Generated by `python scripts/sweep_summary.py` from `{csv_path}`: {len(runs)} runs "
        f"({n_inits} init scales x {n_wds} weight decays x {seeds} seeds), "
        f"test accuracy logged every 100 steps to step {_step(runs['final_step'].max())}.",
        "",
        "Definitions (`eigengrok/analysis/milestones.py`; steps are on the 100-step log grid):",
        "",
        f"- `t_mem`: first step with train accuracy >= {MEM_TRAIN_ACC}",
        f"- `t_onset` (primary): first step with test accuracy >= {ONSET_TEST_ACC}",
        f"- `t_grok`: first step with test accuracy >= {GROK_TEST_ACC}, held for {GROK_SUSTAIN_EVALS} consecutive evaluations",
        "- `gap` = `t_onset` - `t_mem`",
        f"- collapse: test (or train) accuracy falling from >= {COLLAPSE_HIGH} to < {COLLAPSE_LOW}; "
        f"the curve must climb back to >= {COLLAPSE_HIGH} before the next one counts",
        "- secondary onsets (sensitivity only): `onset_2x_chance` (first test accuracy > 2/p) and "
        "`onset_logistic` (exp(mu - 2 sigma) of a logistic fit in log-step, initialized at the 2x-chance crossing)",
        "",
        f"Intervals are {LEVEL:.0%} percentile bootstraps over seeds ({N_BOOT:,} resamples, seed 0). "
        "Pooled-over-weight-decay intervals resample seeds within each weight decay. "
        "Per-run values: `sweep_milestones.csv`; every column below with its interval: `sweep_summary.csv`.",
        "",
        "## Milestones per (init, wd): median [95% CI]",
        "",
        "| init | wd | n | t_mem | t_onset | t_grok (n reached) | gap |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in cells.iterrows():
        L.append(
            f"| {row['init']:g} | {row['wd']:.2f} | {row['n_runs']:.0f} | {_med(row, 't_mem')} | {_med(row, 't_onset')} | "
            f"{_med(row, 't_grok')} ({row['t_grok_n']:.0f}) | {_med(row, 'gap')} |"
        )
    L += [
        "",
        "## Means, final accuracy and collapses per (init, wd)",
        "",
        "Means [95% CI]. Final test accuracy is at the last logged step. Collapses: test-accuracy collapses per run.",
        "",
        f"| init | wd | t_onset mean | gap mean | final test acc mean (min) | final >= {FINAL_ACC_OK} | collapses mean [CI] | collapses min-max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in cells.iterrows():
        fa = f"{row['final_test_acc_mean']:.3f} ({row['final_test_acc_min']:.3f})"
        L.append(
            f"| {row['init']:g} | {row['wd']:.2f} | {_mean(row, 't_onset')} | {_mean(row, 'gap')} | {fa} | "
            f"{row[f'final_test_acc_frac_ge_{FINAL_ACC_OK}']:.0%} | "
            f"{row['n_collapses_mean']:.1f} [{row['n_collapses_mean_ci_lo']:.1f}, {row['n_collapses_mean_ci_hi']:.1f}] | "
            f"{row['n_collapses_min']:.0f}-{row['n_collapses_max']:.0f} |"
        )

    ip = init_pooled
    L += [
        "",
        f"## Init scale: init {alt} against init {INIT_BASELINE} (baseline) at the same weight decay",
        "",
        f"Ratio of init-{alt} to init-{INIT_BASELINE} `t_onset`, 10 seeds each side, [95% CI] from resampling both sides independently:",
        "",
        "| wd | ratio of means | ratio of medians |",
        "|---|---|---|",
    ]
    for _, row in init_wd.iterrows():
        L.append(f"| {row['wd']:.2f} | {_r(row['ratio_mean'])} {_ci(row['ratio_mean_ci'], _r)} | "
                 f"{_r(row['ratio_median'])} {_ci(row['ratio_median_ci'], _r)} |")
    rm, rmed = init_wd["ratio_mean"], init_wd["ratio_median"]
    L += [
        "",
        f"- Ratio of means across the {len(init_wd)} weight decays: range {_r(rm.min())}-{_r(rm.max())}, "
        f"median {_r(rm.median())}; ratio of medians: range {_r(rmed.min())}-{_r(rmed.max())}, median {_r(rmed.median())}. "
        f"Weight decays whose ratio-of-means CI excludes 1: "
        f"{int(sum(lo > 1 or hi < 1 for lo, hi in init_wd['ratio_mean_ci']))} of {len(init_wd)}.",
        f"- Pooled over weight decays, geometric-mean `t_onset` ratio: {_r(ip['onset_gmean_ratio'])} "
        f"{_ci(ip['onset_gmean_ratio_ci'], _r)}.",
        f"- Memorization, pooled median `t_mem`: init {INIT_BASELINE} {_step(ip[f't_mem_median_init{INIT_BASELINE}'])} "
        f"{_ci(ip[f't_mem_median_init{INIT_BASELINE}_ci'])}, init {alt} {_step(ip[f't_mem_median_init{alt}'])} "
        f"{_ci(ip[f't_mem_median_init{alt}_ci'])}; ratio {_r(ip['mem_median_ratio'])} {_ci(ip['mem_median_ratio_ci'], _r)}.",
        f"- Memorization-to-onset gap, pooled median: init {INIT_BASELINE} {_step(ip[f'gap_median_init{INIT_BASELINE}'])} "
        f"{_ci(ip[f'gap_median_init{INIT_BASELINE}_ci'])}, init {alt} {_step(ip[f'gap_median_init{alt}'])} "
        f"{_ci(ip[f'gap_median_init{alt}_ci'])}; ratio {_r(ip['gap_median_ratio'])} {_ci(ip['gap_median_ratio_ci'], _r)}. "
        f"Pooled means: {_step(ip[f'gap_mean_init{INIT_BASELINE}'])} {_ci(ip[f'gap_mean_init{INIT_BASELINE}_ci'])} vs "
        f"{_step(ip[f'gap_mean_init{alt}'])} {_ci(ip[f'gap_mean_init{alt}_ci'])}. Runs with onset before memorization: "
        f"{ip['n_negative_gap']}.",
        f"- Pooled median `t_onset`: init {INIT_BASELINE} {_step(ip[f't_onset_median_init{INIT_BASELINE}'])} "
        f"{_ci(ip[f't_onset_median_init{INIT_BASELINE}_ci'])}, init {alt} {_step(ip[f't_onset_median_init{alt}'])} "
        f"{_ci(ip[f't_onset_median_init{alt}_ci'])}.",
        "",
        f"## Weight decay: wd = {WD_LOW} against wd = {WD_BASELINE} (baseline) at the same init",
        "",
        f"| init | ratio of mean `t_onset` | ratio of median `t_onset` | Spearman(wd, `t_onset`) over all {n_wds} wd |",
        "|---|---|---|---|",
    ]
    for _, row in wd_eff.iterrows():
        L.append(f"| {row['init']:g} | {_r(row['ratio_mean'])} {_ci(row['ratio_mean_ci'], _r)} | "
                 f"{_r(row['ratio_median'])} {_ci(row['ratio_median_ci'], _r)} | "
                 f"{row['spearman_wd_onset']:.2f} (p = {row['spearman_p']:.1e}) |")
    L += [
        "",
        "## Post-grokking collapses",
        "",
        "Collapses per run (distribution over runs); mean [95% CI] resamples seeds within each (init, wd).",
        "",
        "| runs | curve | n | runs with >= 1 | min | 25% | median | 75% | max | mean [CI] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in collapses.iterrows():
        L.append(f"| {row['runs']} | {row['curve']} | {row['n']} | {row['frac_ge_1']:.0%} | {row['min']:.0f} | "
                 f"{row['q25']:.0f} | {row['median']:.0f} | {row['q75']:.0f} | {row['max']:.0f} | "
                 f"{row['mean']:.1f} [{row['mean_ci'][0]:.1f}, {row['mean_ci'][1]:.1f}] |")
    o = onsets
    L += [
        "",
        "## Secondary onset definitions (sensitivity only)",
        "",
        f"Over all {o['n_runs']} runs, against `t_onset`:",
        "",
        "| definition | runs with a value | Spearman with `t_onset` | ratio to `t_onset`: median (10%-90%) |",
        "|---|---|---|---|",
    ]
    for sec in ("onset_2x_chance", "onset_logistic"):
        q = o[f"{sec}_ratio_q"]
        L.append(f"| `{sec}` | {o[f'{sec}_n']} | {o[f'{sec}_spearman']:.2f} | "
                 f"{_r(o[f'{sec}_ratio_median'])} ({_r(q[0])}-{_r(q[1])}) |")
    L += [
        "",
        f"- `onset_2x_chance` fires before memorization (`t_mem`) in {o['n_2x_before_mem']} of {o['n_runs']} runs, "
        f"and at or before step 300 in {o['n_2x_le_300']}.",
        f"- `onset_logistic` and `onset_2x_chance` disagree by more than 1.2x in {o['n_disagree_1p2']} of {o['n_both']} runs.",
        "",
        "Medians [95% CI] per (init, wd):",
        "",
        "| init | wd | onset_2x_chance | onset_2x_chance mean | onset_logistic | 2x chance before t_mem |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in cells.iterrows():
        L.append(f"| {row['init']:g} | {row['wd']:.2f} | {_med(row, 'onset_2x_chance')} | "
                 f"{_mean(row, 'onset_2x_chance')} | {_med(row, 'onset_logistic')} | "
                 f"{row['n_2x_chance_before_mem']:.0f} of {row['n_runs']:.0f} |")
    L.append("")
    path.write_text("\n".join(L))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accuracy-csv", type=Path, default=DEFAULT_ACCURACY_CSV)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    df = pd.read_csv(args.accuracy_csv)
    if df["modulus"].nunique() != 1:
        raise ValueError(f"{args.accuracy_csv} mixes moduli {sorted(df['modulus'].unique())}")
    runs = milestones_table(df)
    missing = runs[runs["t_onset"].isna()]
    if not missing.empty:  # t_onset is the target: a run without one can't silently drop out of a mean
        raise ValueError(f"{len(missing)} run(s) never reach t_onset:\n{missing[['init', 'wd', 'seed']]}")

    cells = cell_summary(runs)
    init_wd, init_pooled = init_effect(runs)
    wd_eff = wd_effect(runs)
    collapses = collapse_stats(runs)
    onsets = onset_definition_stats(runs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "sweep_milestones.csv", index=False)
    cells.to_csv(args.out_dir / "sweep_summary.csv", index=False, float_format="%.6g")
    write_markdown(args.out_dir / "sweep_summary.md", csv_path=args.accuracy_csv, runs=runs, cells=cells,
                   init_wd=init_wd, init_pooled=init_pooled, wd_eff=wd_eff, collapses=collapses, onsets=onsets)
    for name in ("sweep_milestones.csv", "sweep_summary.csv", "sweep_summary.md"):
        print(f"Wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
