# tests/test_checkpoints.py

"""Model-only checkpoints, the Phase 2 checkpoint grid, and pruning a finished run down to it."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eigengrok.analysis.checkpoint_grid import log_grid, onset_window, phase2_steps
from eigengrok.config import ExperimentConfig
from eigengrok.outputs import make_dirs, run_dir
from eigengrok.train.__main__ import main as train_main

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prune_checkpoints", ROOT / "scripts" / "prune_checkpoints.py")
prune_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune_script)

SAVED = [1] + list(range(100, 150_001, 100))  # the sweep's checkpoint steps


def test_log_grid_spans_the_run_evenly_in_log_step():
    grid = log_grid(SAVED)
    assert grid[0] == 1 and grid[-1] == 150_000
    assert set(grid) <= set(SAVED)
    # Nothing is saved between steps 1 and 100, so that stretch collapses onto those two;
    # above it, ~10 points per decade over the 3.2 decades from 100 to 150k.
    assert 28 <= len(grid) <= 36
    above = np.array([s for s in grid if s >= 1000])  # where snapping to 100 is fine enough
    assert np.diff(np.log10(above)).max() < 0.12        # targets are 0.1 decades apart


def test_onset_window():
    assert onset_window(SAVED, 4000) == list(range(2000, 6001, 100))
    assert onset_window(SAVED, float("nan")) == []
    assert set(onset_window(SAVED, 4000)) <= set(phase2_steps(SAVED, 4000))


def test_run_dir_matches_make_dirs(tmp_path):
    cfg = ExperimentConfig(output_dir=str(tmp_path), modulus=91, weight_decay=0.55, initialization_scale=8.0, seed=3)
    assert run_dir(cfg) == make_dirs(cfg)["base"]
    assert run_dir(cfg).as_posix().endswith("modulus=91/depth=3/width=200/init=8/wd=0.55/seed=3")


def test_model_only_checkpoints(tmp_path):
    smoke = ROOT / "configs" / "smoke.yaml"
    train_main(["--config", str(smoke), "--set", f"output_dir={tmp_path}", "optimization_steps=60",
                "log_every=20", "eval_every=20", "checkpoint_every=50", "checkpoint_model_only=true",
                "run_spectral=false"])
    (ckpt,) = tmp_path.rglob("checkpoint_step50.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert set(state) == {"step", "model_state_dict", "config"}


def _fake_run(tmp_path: Path, *, model_only: bool, onset_step: int | None) -> Path:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "figures").mkdir()
    (run / "config_full.json").write_text(json.dumps({"checkpoint_model_only": model_only}))
    steps = np.array(SAVED)
    acc = np.full(len(steps), 0.01) if onset_step is None else np.where(steps >= onset_step, 1.0, 0.01)
    np.savez(run / "figures" / "training_data_x.npz", test_steps=steps, test_accuracies=acc)
    for s in SAVED:
        (run / "checkpoints" / f"checkpoint_step{s}.pt").write_bytes(b"x")
    return run


def _left(run: Path) -> list[int]:
    return sorted(int(p.stem.removeprefix("checkpoint_step")) for p in (run / "checkpoints").glob("*.pt"))


def test_prune_keeps_the_phase2_grid(tmp_path):
    run = _fake_run(tmp_path, model_only=True, onset_step=4000)
    prune_script.prune(run, dry_run=True)
    assert len(_left(run)) == len(SAVED)  # dry run deletes nothing
    prune_script.prune(run)
    assert _left(run) == phase2_steps(SAVED, 4000)
    record = json.loads((run / "checkpoints" / "kept_steps.json").read_text())
    assert record["t_onset"] == 4000 and record["kept_steps"] == _left(run)


def test_prune_keeps_everything_without_onset(tmp_path):
    run = _fake_run(tmp_path, model_only=True, onset_step=None)
    prune_script.prune(run)
    assert _left(run) == SAVED


def test_prune_refuses_full_checkpoint_runs(tmp_path):
    run = _fake_run(tmp_path, model_only=False, onset_step=4000)  # e.g. the original sweep
    with pytest.raises(SystemExit, match="Refusing"):
        prune_script.prune(run)
    assert _left(run) == SAVED
