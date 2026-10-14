# tests/test_config.py

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eigengrok.config import ExperimentConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = ExperimentConfig.from_yaml(path)
    assert isinstance(cfg.torch_dtype, torch.dtype)


def test_overrides_are_coerced_to_field_types(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("task: MODULAR\nlr: 1e-3\n")  # YAML 1.1 reads 1e-3 as a string
    cfg = ExperimentConfig.from_yaml(path, ["weight_decay=1", "seed=3", "run_spectral=false", "optimization_steps=1e4"])
    assert cfg.lr == 1e-3
    assert cfg.weight_decay == 1.0 and isinstance(cfg.weight_decay, float)
    assert cfg.seed == 3
    assert cfg.run_spectral is False
    assert cfg.optimization_steps == 10_000


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("wieght_decay: 0.5\n")
    with pytest.raises(ValueError, match="wieght_decay"):
        ExperimentConfig.from_yaml(path)


@pytest.mark.parametrize("override", ["seed=1.5", "run_spectral=maybe", "dtype=float31", "noequals"])
def test_bad_values_are_rejected(tmp_path, override):
    path = tmp_path / "cfg.yaml"
    path.write_text("task: MODULAR\n")
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(path, [override])


def test_legacy_run_config_loads_leniently():
    """config_full.json files from the existing sweep carry removed fields and a 'torch.float32' dtype string."""
    saved = {"task": "MODULAR", "modulus": 91, "fgsm_epsilon": 0.1, "test_mode": False, "dtype": "torch.float32"}
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(saved)
    cfg = ExperimentConfig.from_dict(saved, strict=False)
    assert cfg.modulus == 91
    assert cfg.torch_dtype is torch.float32


def test_dict_round_trip():
    cfg = ExperimentConfig(task="MNIST", weight_decay=0.3, seed=7)
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg
