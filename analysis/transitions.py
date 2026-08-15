# analysis/transitions.py

"""
Numerical groundwork for the "gate" figure (analysis/make_figures.py):
joins a post-hoc spectral parquet (scripts/spectral_from_checkpoints.py)
with a run's live training-curve npz, establishes the grokking onset two
independent ways, and derives weight-norm-corrected spectral quantities.

Nothing here hardcodes a modulus or a step range -- `modulus` is read off
the run's own config_full.json (or passed explicitly), and the reference
step defaults to the earliest step present in the joined data. Re-running
against a different p's parquet/run_dir needs no code change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REQUIRED_CURVE_KEYS = ("train_steps", "norms", "last_layer_norms", "test_accuracies")

DEFAULT_ACCURACY_CSV = Path("reports/wd_sweep_accuracy_curves.csv")
TRAIN_ACC_LOW_THRESHOLD = 0.9

# Mirrors scripts/build_wd_sweep_accuracy_csv.py's RUN_DIR_RE -- that script
# is what wrote DEFAULT_ACCURACY_CSV in the first place, one row per
# (modulus, depth, width, init, wd, seed, step) parsed straight off each
# run's directory shape. Re-deriving the same key off `run_dir` here (rather
# than off config_full.json) guarantees the lookup matches how the CSV's own
# rows were tagged.
RUN_DIR_META_RE = re.compile(
    r"modulus=(?P<modulus>[\d.]+)/depth=(?P<depth>[\d.]+)/width=(?P<width>[\d.]+)/"
    r"init=(?P<init>[\d.]+)/wd=(?P<wd>[\d.]+)/seed=(?P<seed>\d+)$"
)

# Spectral fields from analysis/spectral_observables.py that carry a
# `<field>_stderr` column in the parquet (scripts/spectral_from_checkpoints.py
# writes one for every dataclass field) and that the gate figure/robustness
# panels read from directly. bulk_edge/conditioning deliberately excluded --
# see "Demoted from headline figures/claims" in reports/spectral_validation.md.
SPECTRAL_FIELDS = ("trace", "top_eig", "effective_rank")


def _find_run_npz(run_dir: Path) -> Path:
    matches = sorted(Path(run_dir).glob("figures/training_data_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No figures/training_data_*.npz under {run_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple figures/training_data_*.npz under {run_dir}, pass one explicitly: {matches}"
        )
    return matches[0]


def load_modulus(run_dir: Path) -> int:
    """Reads `modulus` off the run's config_full.json (written at training time by project_io/dir_making.py)."""
    cfg_path = Path(run_dir) / "config_full.json"
    cfg = json.loads(cfg_path.read_text())
    if "modulus" not in cfg:
        raise KeyError(f"{cfg_path} has no 'modulus' field -- pass --modulus explicitly for a non-MODULAR task.")
    return int(cfg["modulus"])


def load_and_join(parquet_path: Path, run_dir: Path) -> pd.DataFrame:
    """
    Step 1: load the post-hoc spectral parquet and the run's training
    curves (npz), inner-joined on `step`.

    The spectral parquet only has rows at the (sparser) steps a checkpoint
    was requested for; train_steps/test_steps in the npz share one grid
    (see scripts/build_wd_sweep_accuracy_csv.py). A left join validated
    one-to-one catches both a duplicate-step bug in either source and a
    checkpoint step that's off the training log's grid.
    """
    spec = pd.read_parquet(parquet_path)
    npz_path = _find_run_npz(Path(run_dir))
    with np.load(npz_path, allow_pickle=True) as z:
        missing = [k for k in REQUIRED_CURVE_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"{npz_path} is missing expected keys: {missing}")
        curves = pd.DataFrame({
            "step": z["train_steps"],
            "wnorm": z["norms"],
            "ll_norm": z["last_layer_norms"],
            "test_acc": z["test_accuracies"],
        })

    df = spec.merge(curves, on="step", how="left", validate="one_to_one")
    if df["wnorm"].isna().any():
        bad = df.loc[df["wnorm"].isna(), "step"].tolist()
        raise ValueError(
            f"{len(bad)} checkpoint step(s) have no matching row in {npz_path} "
            f"(off the training log's grid): {bad}"
        )
    return df.sort_values("step").reset_index(drop=True)


def load_train_accuracy(
    df: pd.DataFrame, run_dir: Path, *,
    csv_path: Path = DEFAULT_ACCURACY_CSV, low_thresh: float = TRAIN_ACC_LOW_THRESHOLD,
) -> pd.DataFrame:
    """
    Step 1b: joins `train_accuracy` from the wd-sweep accuracy CSV
    (scripts/build_wd_sweep_accuracy_csv.py) onto `df` by step, for this
    run's (modulus, depth, width, init, wd, seed) -- identified off
    `run_dir`'s own path shape (see RUN_DIR_META_RE), the same convention
    that script used to tag the CSV's rows in the first place.

    Adds `train_acc_low` (train_accuracy < `low_thresh`) alongside the raw
    column. Training-accuracy dips mid-run do happen in this sweep --
    AdamW + weight decay can transiently knock a small (train_frac=0.3)
    training set below-threshold even after the run is well past that --
    so a checkpoint landing mid-dip isn't an estimator artifact, but callers
    of the spectral quantities at that step should know it's there.
    """
    m = RUN_DIR_META_RE.search(Path(run_dir).as_posix())
    if not m:
        raise ValueError(
            f"{run_dir} doesn't match the modulus=/depth=/width=/init=/wd=/seed= path shape "
            f"-- can't look up its rows in {csv_path}."
        )
    meta = m.groupdict()

    acc = pd.read_csv(csv_path)
    mask = (
        (acc["modulus"] == int(meta["modulus"]))
        & (acc["depth"] == int(meta["depth"]))
        & (acc["width"] == int(meta["width"]))
        & (acc["init"] == int(meta["init"]))
        & np.isclose(acc["wd"], float(meta["wd"]), atol=1e-9)
        & (acc["seed"] == int(meta["seed"]))
    )
    run_acc = acc.loc[mask, ["step", "train_accuracy"]]
    if run_acc.empty:
        raise ValueError(f"No rows in {csv_path} match {meta} (parsed from {run_dir}).")

    df = df.merge(run_acc, on="step", how="left", validate="one_to_one")
    if df["train_accuracy"].isna().any():
        bad = df.loc[df["train_accuracy"].isna(), "step"].tolist()
        raise ValueError(
            f"{len(bad)} checkpoint step(s) have no train_accuracy row in {csv_path} "
            f"for this run: {bad}"
        )
    df["train_acc_low"] = df["train_accuracy"] < low_thresh
    df = df.sort_values("step").reset_index(drop=True)
    df.attrs["train_acc_low_thresh"] = float(low_thresh)
    return df


@dataclass
class Onset:
    chance: float
    cross_step: int    # first step with test_acc > 2*chance
    fit_step: float     # exp(mu - 2*sigma) of the logistic fit: a shape-independent "2-sigma-before-midpoint" landmark
    mu: float
    sigma: float
    agree: bool          # cross_step and fit_step within `agree_tol` of each other


def _logistic4(logt: np.ndarray, a0: float, ainf: float, mu: float, s: float) -> np.ndarray:
    return a0 + (ainf - a0) / (1 + np.exp(-(logt - mu) / s))


def find_onset(df: pd.DataFrame, modulus: int, *, agree_tol: float = 1.2) -> Onset:
    """
    Step 2: two independent onset estimates, both computed and reported --
    not just the cheap one.

    `cross_step`: first step where test accuracy clears 2x chance. Cheap,
    threshold-based, no fit to go wrong.

    `fit_step`: fits a 4-parameter logistic in log-step, then reports
    exp(mu - 2*sigma) -- two fitted widths before the midpoint, rather than
    an arbitrary accuracy threshold.

    `agree` flags disagreement beyond `agree_tol`x: with a transition that
    can span a fraction of a decade in step count and as few points as one
    checkpoint stride allows, the logistic fit can be poorly conditioned.
    Callers should inspect the curve before trusting `fit_step` alone when
    `agree` is False.
    """
    chance = 1.0 / modulus
    above = df.loc[df["test_acc"] > 2 * chance, "step"]
    if above.empty:
        raise ValueError(f"test_acc never exceeds 2x chance ({2 * chance:.4f}) in this data.")
    cross_step = int(above.iloc[0])

    m = df["test_acc"].notna()
    logt = np.log(df.loc[m, "step"].to_numpy(dtype=float))
    acc = df.loc[m, "test_acc"].to_numpy(dtype=float)
    p0 = [0.0, 1.0, np.log(max(cross_step, 2)), 0.05]
    params, _ = curve_fit(_logistic4, logt, acc, p0=p0, maxfev=20000)
    _, _, mu, sigma = (float(v) for v in params)
    fit_step = float(np.exp(mu - 2 * sigma))

    agree = (1 / agree_tol) <= (fit_step / cross_step) <= agree_tol
    return Onset(chance=chance, cross_step=cross_step, fit_step=fit_step, mu=mu, sigma=sigma, agree=agree)


def add_corrected_quantities(df: pd.DataFrame, ref_step: Optional[int] = None) -> pd.DataFrame:
    """
    Step 3: weight-norm-corrected spectral quantities, all relative to a
    reference step (default: the earliest step in `df`, e.g. step 500 in
    the original spec -- "what a fixed-norm net's curvature would look
    like" starts from wherever the spectral parquet starts).

    `trace_corr`/`top_corr` = trace/top_eig * ||theta||^2, undoing the
    ||theta||^-2 scaling a purely-norm-driven change in curvature would
    produce (loss landscape curvature scales like 1/||theta||^2 under a
    uniform rescaling). `wnorm_pred` is what norm alone predicts for the
    *uncorrected* relative trace -- the check, not the correction.

    Every `_stderr` column here propagates the parquet's own stderr
    linearly through the (deterministic) norm multiplier/reference
    division; the reference step's own sampling uncertainty is neglected,
    consistent with treating step `ref_step` as the fixed comparison point.
    """
    # Positional lookups below (`.iloc[ref]`) require a dense 0..n-1 index --
    # load_and_join() already returns one, but reset defensively in case a
    # caller passes a filtered/re-sorted frame instead.
    df = df.reset_index(drop=True).copy()
    ref = int(df["step"].idxmin()) if ref_step is None else int(df.index[df["step"] == ref_step][0])

    wnorm_ref = df["wnorm"].iloc[ref]
    df["trace_corr"] = df["trace"] * df["wnorm"] ** 2
    df["top_corr"] = df["top_eig"] * df["wnorm"] ** 2
    df["wnorm_pred"] = (wnorm_ref / df["wnorm"]) ** 2

    rel_cols = ["trace", "top_eig", "trace_corr", "top_corr", "effective_rank", "wnorm", "wnorm_pred"]
    for c in rel_cols:
        ref_val = df[c].iloc[ref]
        df[f"{c}_rel"] = df[c] / ref_val
        stderr_col = f"{c}_stderr"
        if stderr_col in df.columns:
            df[f"{c}_rel_stderr"] = df[stderr_col] / abs(ref_val)

    for c in ("trace_corr", "top_corr"):
        base = c.replace("_corr", "")
        stderr_col = f"{base}_stderr"
        if stderr_col in df.columns:
            df[f"{c}_stderr"] = df[stderr_col] * df["wnorm"] ** 2
            ref_val = df[c].iloc[ref]
            df[f"{c}_rel_stderr"] = df[f"{c}_stderr"] / abs(ref_val)

    df.attrs["ref_index"] = ref
    df.attrs["ref_step"] = int(df["step"].iloc[ref])
    return df


def add_robustness_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: two alternative normalizers for the same trace series, so the
    headline ||theta||^-2 correction can be checked rather than trusted:

    `trace_over_top_rel`: trace_rel / top_eig_rel -- a purely internal,
    norm-free scale reference (does the trace move relative to the top
    eigenvalue, independent of any norm story at all).

    `trace_llcorr_rel`: trace * last_layer_norm^2, relative to the same
    reference step -- behaves differently from the whole-network
    correction pre-onset by construction, since total and last-layer norm
    move in opposite directions early in training.

    If the qualitative shape (flat pre-onset, break at the transition)
    survives both, the headline correction isn't a normalization artifact.
    If it flips under either, that's the honest limitation to report.
    """
    df = df.copy()
    ref = df.attrs.get("ref_index")
    if ref is None:
        raise ValueError("Call add_corrected_quantities() before add_robustness_variants().")

    df["trace_over_top_rel"] = df["trace_rel"] / df["top_eig_rel"]

    ll_ref = df["ll_norm"].iloc[ref]
    df["trace_llcorr"] = df["trace"] * df["ll_norm"] ** 2
    df["trace_llcorr_rel"] = df["trace_llcorr"] / df["trace_llcorr"].iloc[ref]

    df.attrs["ll_norm_ref"] = float(ll_ref)
    return df


def _pre_onset_and_smoothness_stats(sub: pd.DataFrame, onset: Onset) -> dict:
    """(a) + (b) of sanity_checks(), computed over whatever subset of rows `sub` is."""
    s: dict = {}

    # (a) does the norm account for the pre-onset trace rise, row by row?
    pre = sub[sub["step"] <= onset.cross_step]
    if len(pre) >= 2:
        resid = np.log(pre["trace_rel"]) - np.log(pre["wnorm_pred"])
        s["pre_onset_log_residual_mean"] = float(resid.mean())
        s["pre_onset_log_residual_std"] = float(resid.std())
        s["pre_onset_log_residual_range"] = float(resid.max() - resid.min())
    else:
        s["pre_onset_log_residual_mean"] = float("nan")
        s["pre_onset_log_residual_std"] = float("nan")
        s["pre_onset_log_residual_range"] = float("nan")

    # (b) is the corrected series smoother than the raw one?
    for c in ["trace_rel", "trace_corr_rel"]:
        s[f"mean_abs_dlog_{c}"] = (
            float(np.abs(np.diff(np.log(sub[c]))).mean()) if len(sub) >= 2 else float("nan")
        )
    return s


def _print_stats(s: dict) -> None:
    print(
        "  pre-onset log-residual (trace_rel vs. ||theta||^-2 prediction): mean %.3f, sd %.3f, range %.3f"
        % (s["pre_onset_log_residual_mean"], s["pre_onset_log_residual_std"], s["pre_onset_log_residual_range"])
    )
    for c in ["trace_rel", "trace_corr_rel"]:
        print(f"  {c} mean |d(log)| per step: {s[f'mean_abs_dlog_{c}']:.4f}")


def sanity_checks(df: pd.DataFrame, onset: Onset, *, exclude_col: str = "train_acc_low") -> dict:
    """
    Step 4: the three checks the spec calls for, run before anyone looks
    at a plot. Prints a human-readable report and returns the same numbers
    for programmatic use (e.g. embedding in the auto-generated figure
    summary -- see analysis/make_figures.py).

    (a) and (b) are reported two ways when `exclude_col` is present (see
    load_train_accuracy()): once over every checkpoint (`report["all"]`),
    once with train-accuracy-dip checkpoints dropped
    (`report["excl_train_acc_low"]`) -- a checkpoint landing mid-dip isn't
    an estimator bug, but it can pull the pre-onset residual/smoothness
    numbers around, so both views get reported rather than one silently
    picked. If `exclude_col` isn't present, both keys hold the same
    all-checkpoints numbers and `n_excluded` is 0.
    """
    ref = df.attrs.get("ref_index")
    if ref is None:
        raise ValueError("Call add_corrected_quantities() before sanity_checks().")

    report: dict = {"n_total": int(len(df))}

    report["all"] = _pre_onset_and_smoothness_stats(df, onset)
    print(f"sanity checks over all {report['n_total']} checkpoint(s):")
    _print_stats(report["all"])

    if exclude_col in df.columns:
        n_excluded = int(df[exclude_col].sum())
        kept = df.loc[~df[exclude_col]]
        report["n_excluded"] = n_excluded
        report["excl_train_acc_low"] = _pre_onset_and_smoothness_stats(kept, onset)
        print(f"sanity checks excluding {n_excluded} checkpoint(s) with train_acc < threshold "
              f"({len(kept)} remain):")
        _print_stats(report["excl_train_acc_low"])
    else:
        report["n_excluded"] = 0
        report["excl_train_acc_low"] = report["all"]

    # (c) is step-to-step scatter above the estimator's own noise floor?
    cols = [c for c in ["step", "top_eig", "top_eig_stderr", "trace", "trace_stderr"] if c in df.columns]
    head = df[cols].head(10)
    report["head"] = head.to_dict(orient="records")
    print(head.to_string(index=False))

    return report
