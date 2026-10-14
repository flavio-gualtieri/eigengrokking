# tests/test_training_options.py

"""Optimizer options (Adam betas, gradient clipping, float64) and the diagnostics logged every log step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eigengrok.config import ExperimentConfig
from eigengrok.train.__main__ import main as train_main
from eigengrok.train.metrics import compute_logit_stats

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
SHORT = ["optimization_steps=40", "log_every=10", "eval_every=10", "checkpoint_every=0", "run_spectral=false"]


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    # A float64 run sets torch's process-wide default dtype; don't leak it into other tests.
    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)


def _run(tmp_path: Path, *overrides: str) -> dict[str, np.ndarray]:
    train_main(["--config", str(SMOKE_CONFIG), "--set", f"output_dir={tmp_path}", *SHORT, *overrides])
    (npz,) = tmp_path.rglob("figures/training_data_*.npz")
    with np.load(npz) as z:
        return {k: z[k] for k in z.files}


def test_optimizer_options_are_parsed(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("task: MODULAR\ngrad_clip: null\n")
    assert ExperimentConfig.from_yaml(path).grad_clip is None
    cfg = ExperimentConfig.from_yaml(path, ["grad_clip=1", "adam_beta2=0.98"])
    assert cfg.grad_clip == 1.0 and isinstance(cfg.grad_clip, float) and cfg.adam_beta2 == 0.98
    assert ExperimentConfig.from_yaml(path, ["grad_clip=none"]).grad_clip is None
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(path, ["grad_clip=often"])
    defaults = ExperimentConfig()
    assert (defaults.adam_beta1, defaults.adam_beta2, defaults.grad_clip) == (0.9, 0.999, None)


def test_betas_reach_the_optimizer(tmp_path):
    _run(tmp_path, "checkpoint_every=20", "adam_beta1=0.8", "adam_beta2=0.98")
    (ckpt,) = tmp_path.rglob("checkpoint_step20.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert state["optimizer_state_dict"]["param_groups"][0]["betas"] == (0.8, 0.98)


def test_diagnostics_are_logged_every_log_step(tmp_path):
    z = _run(tmp_path)
    n = len(z["train_steps"])
    for key in ("grad_norms", "update_norms", "max_abs_logits", "frac_saturated"):
        assert len(z[key]) == n and np.isfinite(z[key]).all(), key
    assert (z["update_norms"] > 0).all() and (z["max_abs_logits"] > 0).all()
    assert ((z["frac_saturated"] >= 0) & (z["frac_saturated"] <= 1)).all()
    assert not list(tmp_path.rglob("*.pt"))  # checkpoint_every=0 writes none


def test_clipping_bounds_the_sgd_step_in_float64(tmp_path):
    # Plain SGD without weight decay steps by exactly lr * (clipped gradient), so
    # the logged update norm must be lr * min(logged pre-clip gradient norm, clip).
    lr, clip = 0.1, 0.05
    z = _run(tmp_path, "optimizer=SGD", "weight_decay=0", f"lr={lr}", f"grad_clip={clip}",
             "dtype=float64", "checkpoint_every=20")
    np.testing.assert_allclose(z["update_norms"], lr * np.minimum(z["grad_norms"], clip), rtol=1e-5)
    assert (z["grad_norms"] > clip).any()  # the clip was active at some logged step

    (ckpt,) = tmp_path.rglob("checkpoint_step20.pt")
    weights = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    assert all(w.dtype == torch.float64 for w in weights.values() if w.is_floating_point())


class _FixedLogits(torch.nn.Module):
    """Returns row x of a fixed logit table for input index x."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits[x.long()]


def test_logit_stats():
    logits = torch.tensor([[100.0, 0.0, 0.0],   # p_max == 1 in float32: saturated
                           [1.0, 0.0, 0.0],     # p_max ~ 0.58
                           [0.0, 30.0, 0.0],    # p_max = 1 - 2e-13: saturated
                           [0.0, 0.0, -5.0]])   # p_max ~ 0.50
    data = torch.utils.data.TensorDataset(torch.arange(4), torch.zeros(4, dtype=torch.long))
    loader = torch.utils.data.DataLoader(data, batch_size=3)  # two batches
    assert compute_logit_stats(_FixedLogits(logits), loader, "cpu") == (100.0, 0.5)
    assert compute_logit_stats(_FixedLogits(logits), loader, "cpu", N=2) == (100.0, 0.5)
    assert compute_logit_stats(_FixedLogits(logits[1:]), loader, "cpu", N=1) == (1.0, 0.0)
