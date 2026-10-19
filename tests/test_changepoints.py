# tests/test_changepoints.py

"""Changepoint detector and lead times (eigengrok/analysis/changepoints.py) on series where the answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigengrok.analysis.changepoints import (
    REFERENCE,
    changepoint_table,
    despike,
    fit_changepoint,
    running_median,
    segmentations,
)

STEPS = np.arange(5000, 15001, 100, dtype=float)  # window [0.5, 1.5] x t_onset for t_onset = 10,000


def _ramp(steps, start, end, high, low):
    """high until `start`, linear down to `low` at `end`, low after."""
    return np.interp(steps, [start, end], [high, low])


def test_running_median_removes_spikes_up_to_two_wide_including_at_the_ends():
    v = np.zeros(12)
    v[[0, 5, 6, 11]] = 9.0
    assert np.allclose(running_median(v, 5), 0.0)


def test_running_median_keeps_a_step_that_lasts():
    v = np.r_[np.zeros(6), np.ones(6)]
    assert np.allclose(running_median(v, 5), v)


def test_despike_removes_spikes_and_leaves_points_away_from_them_as_measured():
    rng = np.random.default_rng(3)
    clean = np.linspace(0, 1, 40) + rng.normal(0, 0.01, 40)
    spikes = [0, 17, 18, 39]
    v = clean.copy()
    v[spikes] += 5.0
    out = despike(v)
    assert np.all(np.abs(out - clean) < 0.1)  # spikes gone; their neighbours move by about one step of the trend
    far = [i for i in range(40) if min(abs(i - j) for j in spikes) > 2]
    assert np.array_equal(out[far], v[far])


def test_segmentation_is_exact_on_a_noiseless_piecewise_line():
    x = np.arange(30, dtype=float)
    y = np.where(x < 10, 0.0, np.where(x < 20, x - 10, 10 - 2 * (x - 20)))
    rss, starts = segmentations(x, y, max_segments=3)[3]
    assert rss == pytest.approx(0.0, abs=1e-9)
    assert starts == [0, 10, 20]


def test_finds_where_a_plateau_starts_to_drop():
    rng = np.random.default_rng(0)
    y = _ramp(STEPS, 9000, 10500, 2.5, 0.0) + rng.normal(0, 0.03, STEPS.size)
    cp = fit_changepoint(STEPS, y)
    assert cp.detected and not cp.at_window_start
    assert abs(cp.step - 9000) <= 200
    assert cp.change < 0


def test_a_drop_with_a_slower_tail_is_dated_by_its_start():
    # steep drop, then a tail a third as steep: one episode, dated where the steep part starts
    y = np.interp(STEPS, [9000, 9600, 11400], [3.0, 1.0, 0.0])
    cp = fit_changepoint(STEPS, y, despike_width=1)
    assert abs(cp.step - 9000) <= 100


def test_a_noisy_straight_line_has_no_changepoint():
    rng = np.random.default_rng(1)
    cp = fit_changepoint(STEPS, 3 - 1e-4 * STEPS + rng.normal(0, 0.05, STEPS.size))
    assert not cp.detected and np.isnan(cp.step)


def test_white_noise_around_a_line_rarely_counts_as_a_change():
    rng = np.random.default_rng(2)
    x = np.arange(60) * 100.0
    hits = [fit_changepoint(x, 1e-4 * x + rng.normal(0, 1, x.size)).detected for _ in range(200)]
    assert np.mean(hits) <= 0.08


def test_a_change_under_way_when_the_window_opens_is_flagged():
    y = _ramp(STEPS, 4000, 8000, 3.0, 0.0)
    cp = fit_changepoint(STEPS, y, despike_width=1)
    assert cp.at_window_start and cp.step == STEPS[0]


def test_direction_ignores_a_steeper_change_the_other_way():
    # a fall at 9000, then a steeper rise at 13000 (a post-grokking collapse in test loss)
    y = _ramp(STEPS, 9000, 10500, 2.5, 0.0) + _ramp(STEPS, 13000, 13300, 0.0, 2.0)
    assert abs(fit_changepoint(STEPS, y, despike_width=1).step - 13000) <= 100
    assert abs(fit_changepoint(STEPS, y, despike_width=1, direction=-1).step - 9000) <= 100


def _run(rank_drop_start: float, t_onset: float = 10_000.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    steps = np.arange(100, 20_001, 100, dtype=float)
    key = {"init": 1.0, "wd": 1.0, "seed": seed}
    curves = pd.DataFrame({**key, "step": steps,
                           "test_loss": np.exp(_ramp(steps, 9000, 10500, 2.5, -3) + rng.normal(0, 0.02, steps.size)),
                           "norm": np.exp(3.7 - 1e-5 * steps + rng.normal(0, 0.01, steps.size))})
    spectra = pd.DataFrame({**key, "step": steps,
                            "effective_rank": np.exp(_ramp(steps, rank_drop_start, rank_drop_start + 1500, 7, 5)
                                                     + rng.normal(0, 0.02, steps.size))})
    milestones = pd.DataFrame([{**key, "t_mem": 200.0, "t_onset": t_onset}])
    return curves, spectra, milestones


def test_lead_time_is_reference_minus_indicator_changepoint():
    table = changepoint_table(*_run(rank_drop_start=8000)).set_index("series")
    assert abs(table.loc[REFERENCE, "changepoint"] - 9000) <= 200
    assert table.loc[REFERENCE, "lead"] == 0
    rank = table.loc["effective_rank"]
    assert abs(rank["lead"] - 1000) <= 300
    assert rank["lead_frac"] == pytest.approx(rank["lead"] / 10_000)
    assert np.isnan(table.loc["norm", "changepoint"])  # a straight line (in log): no change


def test_an_indicator_that_changes_with_the_reference_has_zero_lead():
    table = changepoint_table(*_run(rank_drop_start=9000)).set_index("series")
    assert abs(table.loc["effective_rank", "lead"]) <= 200
