# tests/test_milestones.py

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from eigengrok.analysis.bootstrap import bootstrap_ci, bootstrap_ratio_ci
from eigengrok.analysis.milestones import (
    count_collapses,
    first_step_at_or_above,
    milestones_table,
    onset_2x_chance,
    onset_logistic,
    run_milestones,
    t_grok,
    t_mem,
    t_onset,
)

P = 91
STEPS = np.array([1] + list(range(100, 10_001, 100)), dtype=float)


def _ramp(start: int, stop: int, lo: float = 1 / P, hi: float = 1.0) -> np.ndarray:
    """lo until `start`, linear up to hi at `stop`, hi after."""
    return np.interp(STEPS, [start, stop], [lo, hi])


def test_primary_milestones_of_a_clean_grokking_curve():
    train = _ramp(100, 300)
    test = _ramp(4000, 6000)
    assert t_mem(STEPS, train) == 300           # 0.99 is only reached at the top of the ramp
    assert t_onset(STEPS, test) == 5000         # midpoint of a 4000 -> 6000 ramp
    assert t_grok(STEPS, test) == 5900          # first >= 0.95 (0.9505 at 5900), held to the end


def test_thresholds_are_inclusive():
    acc = np.zeros_like(STEPS)
    acc[10] = 0.5
    acc[20:] = 0.99
    assert t_onset(STEPS, acc) == STEPS[10]
    assert t_mem(STEPS, acc) == STEPS[20]


def test_never_reached_is_nan():
    flat = np.full_like(STEPS, 0.3)
    assert math.isnan(t_mem(STEPS, flat))
    assert math.isnan(t_onset(STEPS, flat))
    assert math.isnan(t_grok(STEPS, flat))


def test_grok_requires_five_consecutive_evaluations():
    acc = np.zeros_like(STEPS)
    acc[10:14] = 0.97           # 4 evals above 0.95, then a collapse: not grokked
    acc[30:35] = 0.96           # exactly 5: grokked here
    assert t_grok(STEPS, acc) == STEPS[30]
    assert first_step_at_or_above(STEPS, acc, 0.95) == STEPS[10]  # the unsustained crossing

    tail = np.zeros_like(STEPS)
    tail[-4:] = 1.0             # only 4 evals left before the end of the run
    assert math.isnan(t_grok(STEPS, tail))


def test_input_order_does_not_matter():
    test = _ramp(4000, 6000)
    perm = np.random.default_rng(0).permutation(len(STEPS))
    assert t_onset(STEPS[perm], test[perm]) == t_onset(STEPS, test)
    assert t_grok(STEPS[perm], test[perm]) == t_grok(STEPS, test)


def test_two_x_chance_is_strict_and_noise_sensitive():
    acc = np.full_like(STEPS, 1 / P)
    acc[1] = 2 / P              # exactly 2x chance: not above it
    acc[3] = 0.03               # a noise blip at step 300
    acc[40:] = 1.0
    assert onset_2x_chance(STEPS, acc, P) == 300
    assert t_onset(STEPS, acc) == STEPS[40]


def test_logistic_onset_recovers_a_known_curve():
    mu, sigma = math.log(5000), 0.1
    acc = 1 / (1 + np.exp(-(np.log(STEPS) - mu) / sigma))
    fit = onset_logistic(STEPS, acc, init_step=3000)
    assert fit.mu == pytest.approx(mu, abs=1e-4)
    assert fit.sigma == pytest.approx(sigma, abs=1e-4)
    assert fit.fit_step == pytest.approx(math.exp(mu - 2 * sigma), rel=1e-3)


def test_count_collapses():
    assert count_collapses([0.0, 0.3, 0.96, 0.4, 0.97, 0.2]) == 2
    assert count_collapses([0.96, 0.7, 0.6, 0.96]) == 0             # dips that stay above 0.5
    assert count_collapses([0.96, 0.8, 0.45, 0.3, 0.1, 0.96]) == 1  # one fall spread over points
    assert count_collapses([0.96, 0.4, 0.8, 0.3]) == 1              # must climb back to 0.95 first
    assert count_collapses([0.4, 0.2, 0.1]) == 0                    # never armed


def test_milestones_table_one_row_per_run():
    rows = []
    for seed, (onset_lo, onset_hi) in enumerate([(4000, 6000), (2000, 3000)]):
        train, test = _ramp(100, 300), _ramp(onset_lo, onset_hi)
        for s, tr, te in zip(STEPS, train, test):
            rows.append(dict(modulus=P, init=1, wd=1.0, seed=seed, step=s,
                             train_accuracy=tr, test_accuracy=te))
    table = milestones_table(pd.DataFrame(rows))
    assert list(table["seed"]) == [0, 1]
    assert list(table["t_onset"]) == [5000, 2500]
    assert list(table["gap"]) == [4700, 2200]
    assert list(table["n_collapses"]) == [0, 0]
    assert list(table["final_test_acc"]) == [1.0, 1.0]
    one = run_milestones(STEPS, _ramp(100, 300), _ramp(4000, 6000), modulus=P)
    assert one["t_onset"] == table["t_onset"].iloc[0]


def test_bootstrap_ci_covers_the_population_value():
    x = np.random.default_rng(1).normal(10.0, 1.0, size=400)
    lo, hi = bootstrap_ci(x, np.mean, n_boot=2000)
    assert lo < 10.0 < hi and hi - lo < 0.5
    assert bootstrap_ci(x, np.mean, n_boot=2000) == (lo, hi)  # seeded: reproducible


def test_bootstrap_drops_nans_and_needs_two_values():
    lo, hi = bootstrap_ci([1.0, np.nan, 3.0, 2.0], np.mean, n_boot=500)
    assert 1.0 <= lo <= hi <= 3.0
    assert all(math.isnan(v) for v in bootstrap_ci([np.nan, 5.0]))


def test_stratified_bootstrap_keeps_cells_apart():
    # Two cells with no within-cell spread: a stratified resample always has
    # the same pooled mean, an unstratified one does not.
    values = np.array([0.0] * 10 + [100.0] * 10)
    strata = np.array([0] * 10 + [1] * 10)
    assert bootstrap_ci(values, np.mean, strata=strata, n_boot=500) == (50.0, 50.0)
    lo, hi = bootstrap_ci(values, np.mean, n_boot=500)
    assert lo < 50.0 < hi


def test_bootstrap_ratio_ci():
    rng = np.random.default_rng(2)
    a, b = rng.normal(20.0, 1.0, 200), rng.normal(10.0, 1.0, 200)
    lo, hi = bootstrap_ratio_ci(a, b, np.mean, n_boot=2000)
    assert lo < 2.0 < hi
