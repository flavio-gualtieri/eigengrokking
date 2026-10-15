# tests/test_forecast.py

"""Phase 3 feature table and evaluation on synthetic sweeps where the answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigengrok.analysis.forecast_eval import oof_predictions, score, split_folds, within_setting_variance
from eigengrok.analysis.forecast_features import (
    SWEEP_CHECKPOINT_STEPS,
    SWEEP_LOG_GRID,
    Cutoff,
    build_features,
    family_columns,
)

INITS, WDS, SEEDS = (1.0, 8.0), (0.45, 0.55, 0.65, 0.75, 0.85, 1.0), range(10)
LAST_STEP = 20_000


def _sweep(signal: bool, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Onset = exp(9.5 - 1.5 wd + 0.2 log2(init) + z) with a per-run offset z. With
    `signal`, the curvature indicator top_eig carries z from the first step;
    otherwise it is noise. The cheap statistics are noise either way.
    """
    rng = np.random.default_rng(seed)
    steps = np.array([s for s in SWEEP_CHECKPOINT_STEPS if s <= LAST_STEP])
    curves, spectra = [], []
    for init in INITS:
        for wd in WDS:
            for s in SEEDS:
                z = rng.normal(0, 0.3)
                onset = float(np.exp(9.5 - 1.5 * wd + 0.2 * np.log2(init) + z))
                mem = 200.0 if init == 1 else 2000.0
                curves.append(pd.DataFrame({
                    "init": init, "wd": wd, "seed": s, "step": steps,
                    "train_accuracy": np.where(steps >= mem, 1.0, 0.1),
                    "test_accuracy": np.where(steps >= onset, 1.0, 0.01),
                    "train_loss": np.exp(rng.normal(0, 0.1, steps.size)),
                    "norm": 50 + rng.normal(0, 1, steps.size),
                    "last_layer_norm": 10 + rng.normal(0, 1, steps.size),
                    "grad_norm": np.exp(rng.normal(0, 0.5, steps.size)),
                }))
                grid = np.array([g for g in SWEEP_LOG_GRID if g <= LAST_STEP])
                level = z if signal else rng.normal(0, 0.3)
                spectra.append(pd.DataFrame({
                    "init": init, "wd": wd, "seed": s, "step": grid,
                    "top_eig": np.exp(3 + level + rng.normal(0, 0.05, grid.size)),
                    "gn_top_eig": np.exp(3 + rng.normal(0, 0.3, grid.size)),
                }))
    return pd.concat(curves, ignore_index=True), pd.concat(spectra, ignore_index=True)


@pytest.fixture(scope="module")
def signal_table():
    return build_features(*_sweep(signal=True), Cutoff("after_mem", 500))


def _gain(table: pd.DataFrame) -> pd.Series:
    preds = {fs: oof_predictions(table, family_columns(table, fs), "ridge", "settings") for fs in ("B0+B1", "B0+B1+C")}
    scores = score(table, preds, baseline="B0+B1", n_boot=300)
    return scores[(scores["features"] == "B0+B1+C") & (scores["metric"] == "r2_within")].iloc[0]


def test_curvature_signal_is_detected_at_the_seed_level(signal_table):
    row = _gain(signal_table)
    assert row["delta"] > 0.5 and row["delta_ci_lo"] > 0.3


def test_noise_curvature_adds_nothing():
    row = _gain(build_features(*_sweep(signal=False), Cutoff("after_mem", 500)))
    assert row["delta_ci_lo"] < 0 < row["delta_ci_hi"] or abs(row["delta"]) < 0.05


def test_hyperparameters_alone_explain_nothing_within_a_setting(signal_table):
    preds = {"B0": oof_predictions(signal_table, family_columns(signal_table, "B0"), "ridge", "settings")}
    s = score(signal_table, preds, n_boot=50).set_index("metric")
    assert abs(s.loc["r2_within", "value"]) < 1e-9
    assert s.loc["r2", "value"] > 0.3  # but they do explain between-setting variance
    v = within_setting_variance(signal_table)
    assert 0 < v["frac_within"] < 1


def test_nothing_after_the_cutoff_is_read():
    curves, spectra = _sweep(signal=True)
    cut = Cutoff("after_mem", 500)
    base = build_features(curves, spectra, cut)
    late_c, late_s = curves.copy(), spectra.copy()
    mem = np.where(late_c["init"] == 1, 200, 2000)
    late_c.loc[late_c["step"] > mem + 500, ["train_loss", "grad_norm"]] = 1e6
    late_s.loc[late_s["step"] > np.where(late_s["init"] == 1, 700, 2500), "top_eig"] = 1e9
    pd.testing.assert_frame_equal(base, build_features(late_c, late_s, cut))


def test_onset_window_checkpoints_are_never_features():
    curves, spectra = _sweep(signal=True)
    cut = Cutoff("after_mem", 500)
    off_grid = sorted(set(range(100, 3000, 100)) - set(SWEEP_LOG_GRID))[:5]
    extra = spectra.groupby(["init", "wd", "seed"]).head(len(off_grid)).copy()
    extra["step"] = np.tile(off_grid, len(extra) // len(off_grid))
    extra["top_eig"] = 1e9
    pd.testing.assert_frame_equal(build_features(curves, spectra, cut),
                                  build_features(curves, pd.concat([spectra, extra]), cut))


def test_absolute_cutoff_censors_memorization_and_drops_finished_runs():
    curves, spectra = _sweep(signal=True)
    table = build_features(curves, spectra, Cutoff("absolute", 1000))
    init8 = table[table["init"] == 8]
    assert np.allclose(init8["B1:t_mem"], np.log(1000)) and (init8["B1:memorized_by_cutoff"] == 0).all()
    late = build_features(curves, spectra, Cutoff("absolute", 6000))
    assert late.attrs["n_dropped"] > 0 and (late["t_onset"] > 6000).all()


def test_splits_keep_settings_apart(signal_table):
    for split in ("settings", "init", "wd_range"):
        for train, test in split_folds(signal_table, split):
            key = signal_table["init"].astype(str) + "/" + signal_table["wd"].astype(str)
            assert not set(key.iloc[train]) & set(key.iloc[test]), split
