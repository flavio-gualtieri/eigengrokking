# eigengrok/analysis/forecast_eval.py

"""
Phase 3 evaluation: out-of-fold predictions of y = ln(t_onset) from a
feature table (eigengrok/analysis/forecast_features.py), under splits that
never let seeds of one hyperparameter setting straddle train and test.

Models: ridge regression (primary; standardised features, median-imputed,
penalty chosen by internal cross-validation) and gradient-boosted trees (a
check for nonlinear signal).

Splits (each run is predicted by a model that never saw its setting):
    settings   leave one (init, wd) setting out at a time (GroupKFold with one group per fold)
    init       train on one init scale, test on the other, both directions
    wd_range   train on wd <= 0.7, test on wd > 0.7

Metrics on the predicted runs:
    r2          1 - SSE / total variance of y
    spearman    rank correlation of prediction and y
    r2_within   R^2 on the deviations from each setting's mean, prediction
                demeaned the same way: how well the model ranks seeds of one
                setting. Hyperparameters alone score exactly 0 here.
Intervals: 95% percentile bootstrap over runs, resampled within settings.
Differences between feature sets are bootstrapped paired (same resample for both).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_BOOT = 2000
WD_SPLIT = 0.7
METRICS = ("r2", "spearman", "r2_within")


def make_model(name: str):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 4, 15)))
    if name == "gbt":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=10, l2_regularization=1.0, random_state=0)
    raise ValueError(f"Unknown model {name!r}")


def _setting(table: pd.DataFrame) -> np.ndarray:
    return (table["init"].astype(str) + "/" + table["wd"].astype(str)).to_numpy()


def split_folds(table: pd.DataFrame, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(table))
    if split == "settings":
        groups = _setting(table)
        return list(GroupKFold(n_splits=len(set(groups))).split(idx, groups=groups))
    if split == "init":
        inits = sorted(table["init"].unique())
        return [(idx[table["init"].to_numpy() != i], idx[table["init"].to_numpy() == i]) for i in inits]
    if split == "wd_range":
        low = table["wd"].to_numpy() <= WD_SPLIT
        return [(idx[low], idx[~low])]
    raise ValueError(f"Unknown split {split!r}")


def oof_predictions(table: pd.DataFrame, columns: list[str], model: str, split: str) -> np.ndarray:
    """Out-of-fold predictions; NaN for runs no fold tests (the training side of wd_range)."""
    X, y = table[columns].to_numpy(dtype=float), table["y"].to_numpy(dtype=float)
    pred = np.full(len(table), np.nan)
    for train, test in split_folds(table, split):
        if len(train) < 5 or len(test) == 0:
            continue
        pred[test] = make_model(model).fit(X[train], y[train]).predict(X[test])
    return pred


def _metric_fns(settings: np.ndarray) -> dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]]:
    def r2(y, p, s):
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    def spearman(y, p, s):
        return float(spearmanr(y, p)[0])

    def r2_within(y, p, s):
        dy, dp = y.copy(), p.copy()
        for g in np.unique(s):
            m = s == g
            dy[m] -= y[m].mean()
            dp[m] -= p[m].mean()
        denom = np.sum(dy ** 2)
        return 1 - np.sum((dy - dp) ** 2) / denom if denom > 0 else np.nan

    return {"r2": r2, "spearman": spearman, "r2_within": r2_within}


def score(table: pd.DataFrame, preds: dict[str, np.ndarray], *, n_boot: int = N_BOOT, seed: int = 0,
          baseline: str | None = None) -> pd.DataFrame:
    """
    Metrics with 95% bootstrap intervals for each feature set's predictions,
    and, if `baseline` is given, each set's difference from it (paired).
    Only runs every feature set predicted are scored, so all are compared on the same runs.
    """
    have = np.all([np.isfinite(p) for p in preds.values()], axis=0)
    if have.sum() < 3:  # e.g. the init split before both init scales are in the data
        return pd.DataFrame([{"features": name, "metric": metric, "value": np.nan, "n_runs": int(have.sum()),
                              "ci_lo": np.nan, "ci_hi": np.nan} for name in preds for metric in METRICS])
    y, s = table["y"].to_numpy(dtype=float)[have], _setting(table)[have]
    P = {k: p[have] for k, p in preds.items()}
    fns = _metric_fns(s)

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(s == g) for g in np.unique(s)]
    resamples = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(n_boot)]

    rows = []
    for name, p in P.items():
        for metric, fn in fns.items():
            point = fn(y, p, s)
            boots = np.array([fn(y[r], p[r], s[r]) for r in resamples])
            row = {"features": name, "metric": metric, "value": point, "n_runs": len(y),
                   "ci_lo": np.nanquantile(boots, 0.025), "ci_hi": np.nanquantile(boots, 0.975)}
            if baseline is not None and name != baseline:
                b = P[baseline]
                d_boots = boots - np.array([fn(y[r], b[r], s[r]) for r in resamples])
                row.update(delta_vs=baseline, delta=point - fn(y, b, s),
                           delta_ci_lo=np.nanquantile(d_boots, 0.025), delta_ci_hi=np.nanquantile(d_boots, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def within_setting_variance(table: pd.DataFrame) -> dict[str, float]:
    """Total variance of y and the part that is between seeds of one setting (pooled within-setting variance)."""
    y, s = table["y"].to_numpy(dtype=float), _setting(table)
    within = sum(np.sum((y[s == g] - y[s == g].mean()) ** 2) for g in np.unique(s))
    total = np.sum((y - y.mean()) ** 2)
    return {"var_total": total / len(y), "var_within": within / len(y), "frac_within": within / total}
