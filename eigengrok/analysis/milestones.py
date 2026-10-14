# eigengrok/analysis/milestones.py

"""
Grokking milestones of one run, read off its logged accuracy curves. This is
the single definition of every milestone: each analysis script imports them
from here rather than re-deriving its own threshold.

A milestone is a logged step, so it is resolved only to the evaluation grid
(every 100 steps in the p=91 sweep). A milestone a run never reaches is NaN,
never a silent default; callers decide whether that is an error.

Primary:

    t_mem    first step with train accuracy >= 0.99 (memorization)
    t_onset  first step with test accuracy >= 0.5 (generalization onset).
             THE target: every forecast and every sweep-level figure uses it.
    t_grok   first step with test accuracy >= 0.95 that stays >= 0.95 for 5
             consecutive evaluations (that step and the next four)

Secondary, for sensitivity analysis only (never a headline number):

    onset_2x_chance  first step with test accuracy > 2/p
    onset_logistic   exp(mu - 2 sigma) of a 4-parameter logistic fit in log-step

Instability:

    count_collapses  falls from >= 0.95 to < 0.5 (test or train accuracy)

`milestones_table` applies all of these to the long-format sweep CSV
(scripts/build_wd_sweep_accuracy_csv.py), one row per run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

MEM_TRAIN_ACC = 0.99
ONSET_TEST_ACC = 0.5
GROK_TEST_ACC = 0.95
GROK_SUSTAIN_EVALS = 5
COLLAPSE_HIGH = 0.95
COLLAPSE_LOW = 0.5

RUN_KEYS = ("init", "wd", "seed")


def _sorted(steps, values) -> tuple[np.ndarray, np.ndarray]:
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    if steps.shape != values.shape:
        raise ValueError(f"steps {steps.shape} and values {values.shape} differ in shape")
    order = np.argsort(steps, kind="stable")
    return steps[order], values[order]


def _first_step(steps: np.ndarray, hit: np.ndarray) -> float:
    idx = np.flatnonzero(hit)
    return float(steps[idx[0]]) if idx.size else float("nan")


def first_step_at_or_above(steps, values, threshold: float, *, sustain: int = 1) -> float:
    """
    First step at which `values` >= `threshold` and stays there for `sustain`
    consecutive logged points (that one included). NaN if that never happens,
    including when fewer than `sustain` points remain at the end of the run.
    """
    if sustain < 1:
        raise ValueError("sustain must be >= 1")
    steps, values = _sorted(steps, values)
    ok = values >= threshold  # NaN compares False
    if sustain > 1:
        if ok.size < sustain:
            return float("nan")
        window = np.convolve(ok.astype(int), np.ones(sustain, dtype=int), mode="valid")
        ok = np.concatenate([window == sustain, np.zeros(sustain - 1, dtype=bool)])
    return _first_step(steps, ok)


def t_mem(steps, train_acc) -> float:
    """Memorization: first step with train accuracy >= 0.99."""
    return first_step_at_or_above(steps, train_acc, MEM_TRAIN_ACC)


def t_onset(steps, test_acc) -> float:
    """
    Generalization onset (PRIMARY): first step with test accuracy >= 0.5.
    Far enough from chance (1/p ~ 0.011 at p=91) that pre-training softmax
    noise never crosses it, and every run of the p=91 sweep reaches it.
    """
    return first_step_at_or_above(steps, test_acc, ONSET_TEST_ACC)


def t_grok(steps, test_acc) -> float:
    """
    Grokked: first step with test accuracy >= 0.95 that holds for 5
    consecutive evaluations (that step plus the next four). The sustain
    requirement skips brief excursions above 0.95 that collapse straight back.
    """
    return first_step_at_or_above(steps, test_acc, GROK_TEST_ACC, sustain=GROK_SUSTAIN_EVALS)


def onset_2x_chance(steps, test_acc, modulus: int) -> float:
    """
    SECONDARY onset: first step with test accuracy > 2/p (strictly above, as
    the original gate figure defined it). Noise-dominated across seeds: some
    runs clear it within the first few hundred steps, before memorization,
    then fall back to chance until the real transition. Sensitivity analysis only.
    """
    steps, test_acc = _sorted(steps, test_acc)
    return _first_step(steps, test_acc > 2.0 / modulus)


@dataclass
class LogisticOnset:
    fit_step: float  # exp(mu - 2*sigma): two fitted widths before the midpoint
    mu: float
    sigma: float


def _logistic4(logt: np.ndarray, a0: float, ainf: float, mu: float, s: float) -> np.ndarray:
    return a0 + (ainf - a0) / (1 + np.exp(-(logt - mu) / s))


def onset_logistic(steps, test_acc, *, init_step: float) -> LogisticOnset:
    """
    SECONDARY onset: fits a + (b - a) / (1 + exp(-(log t - mu) / sigma)) to
    test accuracy against log-step and reports exp(mu - 2 sigma), a landmark
    two fitted widths before the midpoint. `init_step` seeds the midpoint
    guess (the gate figure uses the 2x-chance crossing). Poorly conditioned
    when the rise spans few logged points, and pulled around by post-grok
    collapses; all-NaN if the fit does not converge. Sensitivity analysis only.
    """
    steps, test_acc = _sorted(steps, test_acc)
    keep = np.isfinite(test_acc) & (steps > 0)
    logt, acc = np.log(steps[keep]), test_acc[keep]
    guess = init_step if np.isfinite(init_step) else float(np.exp(np.median(logt)))
    p0 = [0.0, 1.0, np.log(max(guess, 2)), 0.05]
    try:
        # exp overflow just saturates the sigmoid; the covariance estimate is never used
        with np.errstate(over="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(_logistic4, logt, acc, p0=p0, maxfev=20000)
    except RuntimeError:
        return LogisticOnset(fit_step=float("nan"), mu=float("nan"), sigma=float("nan"))
    _, _, mu, sigma = (float(v) for v in params)
    return LogisticOnset(fit_step=float(np.exp(mu - 2 * sigma)), mu=mu, sigma=sigma)


def count_collapses(values, *, high: float = COLLAPSE_HIGH, low: float = COLLAPSE_LOW) -> int:
    """
    Number of collapses in an accuracy curve (in step order): falls below
    `low` after having been at or above `high`. The curve must climb back to
    `high` before the next fall counts, so a fall that spans several logged
    points is one collapse, and dips that stay above `low` count as none.
    """
    armed, n = False, 0
    for v in np.asarray(values, dtype=float):
        if v >= high:
            armed = True
        elif v < low and armed:
            n += 1
            armed = False
    return n


def run_milestones(steps, train_acc, test_acc, *, modulus: int, secondary: bool = True) -> dict[str, float]:
    """Every milestone of one run, plus final test accuracy and collapse counts."""
    steps_s, train_s = _sorted(steps, train_acc)
    _, test_s = _sorted(steps, test_acc)
    out = {
        "t_mem": t_mem(steps_s, train_s),
        "t_onset": t_onset(steps_s, test_s),
        "t_grok": t_grok(steps_s, test_s),
    }
    out["gap"] = out["t_onset"] - out["t_mem"]
    out["final_step"] = float(steps_s[-1])
    out["final_test_acc"] = float(test_s[-1])
    out["n_collapses"] = count_collapses(test_s)
    out["n_train_collapses"] = count_collapses(train_s)
    if secondary:
        out["onset_2x_chance"] = onset_2x_chance(steps_s, test_s, modulus)
        out["onset_logistic"] = onset_logistic(steps_s, test_s, init_step=out["onset_2x_chance"]).fit_step
    return out


def milestones_table(
        df: pd.DataFrame,
        *,
        by: Sequence[str] = RUN_KEYS,
        secondary: bool = True,
) -> pd.DataFrame:
    """
    One row per run of a long-format accuracy table (columns `by`, modulus,
    step, train_accuracy, test_accuracy, as in reports/wd_sweep_accuracy_curves.csv)
    with every milestone of `run_milestones`.
    """
    rows = []
    for key, g in df.groupby(list(by), sort=True):
        moduli = g["modulus"].unique()
        if len(moduli) != 1:
            raise ValueError(f"Run {dict(zip(by, key))} mixes moduli {moduli}")
        m = run_milestones(g["step"], g["train_accuracy"], g["test_accuracy"],
                           modulus=int(moduli[0]), secondary=secondary)
        rows.append({**dict(zip(by, key)), **m})
    return pd.DataFrame(rows)
