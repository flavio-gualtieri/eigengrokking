#!/usr/bin/env python3
# scripts/forecast_benchmark.py

"""
Phase 3: from the first part of training, does curvature predict
ln(t_onset) better than the hyperparameters and cheap training statistics?

    python scripts/forecast_benchmark.py

Reads the rerun sweep's curves (scripts/build_wd_sweep_accuracy_csv.py
output) and, if present, its per-run spectra (scripts/run_spectral_sweep_array.sh).
Features: eigengrok/analysis/forecast_features.py. Models, splits, metrics
and intervals: eigengrok/analysis/forecast_eval.py. Without spectra it runs
the baselines only (B0, B0+B1).

Writes (see --out-dir, default reports/):
    forecast_results.csv       every (cutoff, model, split, feature set, metric), with 95% CIs
    forecast_predictions.csv   out-of-fold predictions of the headline setting (primary cutoff, ridge, settings split)
    forecast_summary.md        the headline comparison and the robustness tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eigengrok.analysis.forecast_eval import N_BOOT, oof_predictions, score, within_setting_variance
from eigengrok.analysis.forecast_features import (
    TARGETS,
    Cutoff,
    build_features,
    family_columns,
    load_curves,
    load_spectra,
)

PRIMARY_CUTOFF = Cutoff("after_mem", 500)
CUTOFFS = [PRIMARY_CUTOFF, Cutoff("after_mem", 1000), Cutoff("absolute", 1000), Cutoff("absolute", 2000)]
MODELS = ("ridge", "gbt")
SPLITS = ("settings", "init", "wd_range")
BASELINE = "B0+B1"


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    sets = {"B0": family_columns(table, "B0"), "B0+B1": family_columns(table, "B0+B1")}
    curvature = family_columns(table, "C")
    if curvature:
        sets["B0+B1+C"] = sets["B0+B1"] + curvature
        sets["C"] = curvature
        sets["B0+B1+C(H only)"] = sets["B0+B1"] + [c for c in curvature if not c.startswith("C:gn_")]
        sets["B0+B1+C(G only)"] = sets["B0+B1"] + [c for c in curvature if c.startswith("C:gn_")]
    return sets


def run(curves: pd.DataFrame, spectra: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    results, info, headline = [], {}, None
    for cutoff in CUTOFFS:
        table = build_features(curves, spectra, cutoff)
        info[cutoff.label] = {"n_runs": len(table), "n_dropped": table.attrs["n_dropped"],
                              **within_setting_variance(table)}
        sets = feature_sets(table)
        for model in MODELS:
            for split in SPLITS:
                preds = {name: oof_predictions(table, cols, model, split) for name, cols in sets.items()}
                s = score(table, preds, n_boot=n_boot, baseline=BASELINE)
                if cutoff == PRIMARY_CUTOFF and model == "ridge" and split == "settings":
                    headline = table[["init", "wd", "seed", "t_onset", "y"]].assign(
                        **{f"pred:{name}": p for name, p in preds.items()})
                results.append(s.assign(cutoff=cutoff.label, model=model, split=split))
                print(f"{cutoff.label:12s} {model:5s} {split:8s} done")

    # Robustness to the onset definition: the secondary targets at the primary cutoff, settings split.
    for target in TARGETS[1:]:
        table = build_features(curves, spectra, PRIMARY_CUTOFF, target=target)
        info[f"{PRIMARY_CUTOFF.label}, target {target}"] = {"n_runs": len(table), "n_dropped": table.attrs["n_dropped"],
                                                           **within_setting_variance(table)}
        sets = feature_sets(table)
        for model in MODELS:
            preds = {name: oof_predictions(table, cols, model, "settings") for name, cols in sets.items()}
            s = score(table, preds, n_boot=n_boot, baseline=BASELINE)
            results.append(s.assign(cutoff=PRIMARY_CUTOFF.label, model=model, split="settings", target=target))
            print(f"target {target:16s} {model:5s} done")
    # Sensitivity to clipping held-out features to the training range (added after the first audit run, to stop
    # the init split extrapolating): the headline comparison without it.
    table = build_features(curves, spectra, PRIMARY_CUTOFF)
    sets = feature_sets(table)
    for model in MODELS:
        for split in ("settings", "wd_range"):
            preds = {name: oof_predictions(table, cols, model, split, clip=False) for name, cols in sets.items()}
            s = score(table, preds, n_boot=n_boot, baseline=BASELINE)
            results.append(s.assign(cutoff=PRIMARY_CUTOFF.label, model=model, split=split, target="t_onset",
                                    clipping="off"))
            print(f"no clipping {model:5s} {split:8s} done")
    results = pd.concat(results, ignore_index=True)
    results["target"] = results["target"].fillna("t_onset") if "target" in results else "t_onset"
    results["clipping"] = results["clipping"].fillna("on")
    return results, info, headline


def _ci(lo, hi, fmt="{:.2f}") -> str:
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def markdown(results: pd.DataFrame, info: dict, has_spectra: bool, n_runs_total: int) -> str:
    unclipped = results[results["clipping"] == "off"] if "clipping" in results else results.iloc[0:0]
    L = ["# Forecasting ln(t_onset) (Phase 3)", "",
         "Generated by `python scripts/forecast_benchmark.py`. Target: ln(t_onset), t_onset = first step with "
         "test accuracy >= 0.5. Every run is predicted by a model that never saw its (init, wd) setting. "
         "Intervals: 95% bootstrap over runs, resampled within settings; differences are paired. "
         "`r2_within`: R^2 on deviations from each setting's mean (seed-level prediction); hyperparameters "
         "alone score 0 there by construction.", ""]
    if not has_spectra:
        L += ["**Baselines only: no curvature spectra were found, so C is not evaluated yet.**", ""]
    L += [f"Runs: {n_runs_total} in the sweep data.", "",
          "| cutoff | runs evaluated | dropped (onset already reached) | var(ln t_onset) | share within settings |",
          "|---|---|---|---|---|"]
    for label, d in info.items():
        L.append(f"| {label} | {d['n_runs']} | {d['n_dropped']} | {d['var_total']:.3f} | {d['frac_within']:.0%} |")

    results = results[results["clipping"] == "on"].copy() if "clipping" in results else results
    head = results[(results["cutoff"] == PRIMARY_CUTOFF.label) & (results["model"] == "ridge")
                   & (results["target"] == "t_onset")]
    L += ["", f"## Headline: ridge, cutoff {PRIMARY_CUTOFF.label}", "",
          "| split | features | R^2 [CI] | Spearman [CI] | R^2 within settings [CI] | gain over B0+B1 in R^2 within [CI] |",
          "|---|---|---|---|---|---|"]
    for split in SPLITS:
        h = head[head["split"] == split]
        if h["value"].isna().all():
            L.append(f"| {split} | (not evaluable with the runs available) | | | | |")
            continue
        for fs in h["features"].unique():
            r = h[h["features"] == fs].set_index("metric")
            gain = "--"
            if fs not in ("B0", BASELINE) and np.isfinite(r.loc["r2_within"].get("delta", np.nan)):
                w = r.loc["r2_within"]
                gain = f"{w['delta']:+.3f} {_ci(w['delta_ci_lo'], w['delta_ci_hi'], '{:+.3f}')}"
            L.append(f"| {split} | {fs} | {r.loc['r2', 'value']:.2f} {_ci(r.loc['r2', 'ci_lo'], r.loc['r2', 'ci_hi'])} | "
                     f"{r.loc['spearman', 'value']:.2f} {_ci(r.loc['spearman', 'ci_lo'], r.loc['spearman', 'ci_hi'])} | "
                     f"{r.loc['r2_within', 'value']:.2f} {_ci(r.loc['r2_within', 'ci_lo'], r.loc['r2_within', 'ci_hi'])} | {gain} |")

    L += ["", "## Robustness: every cutoff and model, settings split", "",
          "| cutoff | model | features | R^2 | R^2 within | gain in R^2 within over B0+B1 [CI] |", "|---|---|---|---|---|---|"]
    rob = results[(results["split"] == "settings") & (results["target"] == "t_onset")]
    for (cut, model), g in rob.groupby(["cutoff", "model"], sort=False):
        if g["value"].isna().all():
            continue
        for fs in g["features"].unique():
            r = g[g["features"] == fs].set_index("metric")
            w = r.loc["r2_within"]
            gain = (f"{w['delta']:+.3f} {_ci(w['delta_ci_lo'], w['delta_ci_hi'], '{:+.3f}')}"
                    if fs not in ("B0", BASELINE) and isinstance(w.get("delta_vs"), str) else "--")
            L.append(f"| {cut} | {model} | {fs} | {r.loc['r2', 'value']:.2f} | {w['value']:.2f} | {gain} |")

    L += ["", f"## Robustness: onset definition, cutoff {PRIMARY_CUTOFF.label}, settings split", "",
          "Secondary onsets (eigengrok/analysis/milestones.py) as the target. A run whose onset under that "
          "definition is already at or before the cutoff is dropped (counts in the first table).", "",
          "| target | model | features | R^2 | R^2 within | gain in R^2 within over B0+B1 [CI] |", "|---|---|---|---|---|---|"]
    sec = results[(results["split"] == "settings") & (results["cutoff"] == PRIMARY_CUTOFF.label)]
    for (target, model), g in sec.groupby(["target", "model"], sort=False):
        if g["value"].isna().all():
            L.append(f"| {target} | {model} | (not evaluable with the runs available) | | | |")
            continue
        for fs in g["features"].unique():
            r = g[g["features"] == fs].set_index("metric")
            w = r.loc["r2_within"]
            gain = (f"{w['delta']:+.3f} {_ci(w['delta_ci_lo'], w['delta_ci_hi'], '{:+.3f}')}"
                    if fs not in ("B0", BASELINE) and isinstance(w.get("delta_vs"), str) else "--")
            L.append(f"| {target} | {model} | {fs} | {r.loc['r2', 'value']:.2f} | {w['value']:.2f} | {gain} |")

    if len(unclipped):
        L += ["", f"## Sensitivity: without clipping held-out features to the training range, cutoff {PRIMARY_CUTOFF.label}", "",
              "Every other table clips each test fold's features to the range of its training fold (added after the "
              "first full run, to stop the init split extrapolating). The same comparison without it:", "",
              "| split | model | features | R^2 | R^2 within | gain in R^2 within over B0+B1 [CI] |", "|---|---|---|---|---|---|"]
        for (split, model), g in unclipped.groupby(["split", "model"], sort=False):
            for fs in g["features"].unique():
                r = g[g["features"] == fs].set_index("metric")
                w = r.loc["r2_within"]
                gain = (f"{w['delta']:+.3f} {_ci(w['delta_ci_lo'], w['delta_ci_hi'], '{:+.3f}')}"
                        if fs not in ("B0", BASELINE) and isinstance(w.get("delta_vs"), str) else "--")
                L.append(f"| {split} | {model} | {fs} | {r.loc['r2', 'value']:.2f} | {w['value']:.2f} | {gain} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accuracy-csv", type=Path, default=Path("reports/sweep_beta2_0.98_accuracy_curves.csv.gz"))
    parser.add_argument("--stats-csv", type=Path, default=Path("reports/sweep_beta2_0.98_training_stats.csv.gz"))
    parser.add_argument("--spectra-dir", type=Path, default=Path("reports/spectra_sweep"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    curves = load_curves(args.accuracy_csv, args.stats_csv)
    spectra = load_spectra(args.spectra_dir) if args.spectra_dir.exists() else None
    has_spectra = spectra is not None and len(spectra) > 0
    results, info, headline = run(curves, spectra if has_spectra else None, args.n_boot)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "forecast_results.csv", index=False, float_format="%.6g")
    headline.to_csv(args.out_dir / "forecast_predictions.csv", index=False, float_format="%.6g")
    n_runs = curves.groupby(["init", "wd", "seed"]).ngroups
    (args.out_dir / "forecast_summary.md").write_text(markdown(results, info, has_spectra, n_runs))
    print(f"Wrote {args.out_dir / 'forecast_results.csv'} and {args.out_dir / 'forecast_summary.md'}")


if __name__ == "__main__":
    main()
