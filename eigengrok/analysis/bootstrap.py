# eigengrok/analysis/bootstrap.py

"""
Percentile bootstrap confidence intervals over runs.

Every interval resamples runs (seeds) with replacement. When a statistic
pools runs from several design cells (e.g. every weight decay at one init
scale), `strata` resamples within each cell, so each resample keeps the
design's cell sizes and only the seed-to-seed variation is bootstrapped.
NaNs are dropped before resampling. Fixed `seed`, so intervals are reproducible.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Statistic = Callable[..., np.ndarray]  # called as stat(x, axis=1) on [n_boot, n] resamples

N_BOOT = 10_000
LEVEL = 0.95


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _resample(groups: Sequence[np.ndarray], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """[n_boot, sum of group sizes]: every group resampled with replacement, side by side."""
    return np.concatenate([g[rng.integers(0, len(g), size=(n_boot, len(g)))] for g in groups], axis=1)


def _groups(values, strata) -> list[np.ndarray]:
    values = np.asarray(values, dtype=float)
    if strata is None:
        return [_finite(values)]
    strata = np.asarray(strata)
    return [g for g in (_finite(values[strata == s]) for s in np.unique(strata)) if len(g)]


def _interval(boots: np.ndarray, level: float) -> tuple[float, float]:
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def bootstrap_ci(values, stat: Statistic = np.median, *, strata=None, n_boot: int = N_BOOT,
                 level: float = LEVEL, seed: int = 0) -> tuple[float, float]:
    """(lo, hi) percentile interval of `stat` over `values`, optionally stratified."""
    groups = _groups(values, strata)
    if sum(len(g) for g in groups) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    return _interval(stat(_resample(groups, n_boot, rng), axis=1), level)


def bootstrap_ratio_ci(numerator, denominator, stat: Statistic = np.mean, *, strata_num=None,
                       strata_den=None, n_boot: int = N_BOOT, level: float = LEVEL,
                       seed: int = 0) -> tuple[float, float]:
    """
    (lo, hi) interval of stat(numerator) / stat(denominator), two independent
    sets of runs, each resampled on its own (optionally stratified).
    """
    num, den = _groups(numerator, strata_num), _groups(denominator, strata_den)
    if sum(len(g) for g in num) < 2 or sum(len(g) for g in den) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ratio = stat(_resample(num, n_boot, rng), axis=1) / stat(_resample(den, n_boot, rng), axis=1)
    return _interval(ratio, level)
