# tests/test_smoke.py

"""
End-to-end checks of `python -m eigengrok.train`: a shortened version of
configs/smoke.yaml must write every output, and a fixed seed must reproduce
the same training curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eigengrok.config import ExperimentConfig
from eigengrok.hessian.spectrum import STAT_NAMES
from eigengrok.train.__main__ import main as train_main

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def _run(tmp_path: Path, *overrides: str) -> Path:
    train_main(["--config", str(SMOKE_CONFIG), "--set", f"output_dir={tmp_path}", *overrides])
    run_dirs = [p.parent for p in tmp_path.rglob("config_full.json")]
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]


def _training_data(run_dir: Path) -> dict[str, np.ndarray]:
    (npz_path,) = run_dir.glob("figures/training_data_*.npz")
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}


def test_smoke_run_writes_every_output(tmp_path):
    overrides = ["optimization_steps=120", "log_every=40", "eval_every=40", "checkpoint_every=50"]
    run_dir = _run(tmp_path, *overrides)

    saved = json.loads((run_dir / "config_full.json").read_text())
    assert ExperimentConfig.from_dict(saved) == ExperimentConfig.from_yaml(
        SMOKE_CONFIG, [f"output_dir={tmp_path}", *overrides])
    assert (run_dir / "identity.json").exists()

    ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    assert ckpts == ["checkpoint_step1.pt", "checkpoint_step100.pt", "checkpoint_step50.pt"]

    data = _training_data(run_dir)
    assert list(data["train_steps"]) == [1, 40, 80]
    assert list(data["test_steps"]) == [1, 40, 80]

    # Spectral snapshots were taken and are finite.
    n_snapshots = len(data["eig_steps"])
    assert n_snapshots > 0
    for name in STAT_NAMES:
        assert len(data[name]) == n_snapshots, name
        assert np.isfinite(data[name]).all(), name
    assert len(list((run_dir / "spectral_snapshots").glob("*.png"))) == n_snapshots
    assert len(list((run_dir / "figures").glob("training_curve_*.png"))) > 0


def test_same_seed_reproduces_training_curve(tmp_path):
    overrides = ["optimization_steps=60", "log_every=20", "eval_every=20", "checkpoint_every=1000",
                 "run_spectral=false"]
    first = _training_data(_run(tmp_path / "a", *overrides))
    second = _training_data(_run(tmp_path / "b", *overrides))
    for key in ("train_losses", "test_losses", "norms"):
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)
