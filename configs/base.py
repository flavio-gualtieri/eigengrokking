# configs/base.py

from __future__ import annotations

from dataclasses import dataclass

import torch
import os

@dataclass(frozen=True)
class ExperimentConfig:
    # Admin
    run_spectral: bool = True
    test_mode: bool = False  # If True, saves to a TEST subdir and uses a simplified config for quick runs
    task: str = "MOD"

    # Data
    data_directory: str = os.environ.get("DATA_DIR", "./data")
    train_points: int = 2000
    batch_size: int = 300

    # Modular arithmetic task
    modulus: int = 97
    train_frac: float = 0.3

    # Model
    input_dim: int = 784
    output_dim: int = 10
    depth: int = 3
    width: int = 200
    num_heads: int = 4
    activation: str = "ReLU"
    initialization_scale: float = 1.0

    # Spectral analysis
    spectral_m: int = 100              # Lanczos steps per probe -- see reports/spectral_validation.md
    spectral_k: int = 100              # number of Rademacher probes (SLQ) -- k=16 under-converged trace (~17-19% error); k=100 gets ~5%
    spectral_sigma: float = 0.01      # KDE bandwidth for the plotted density
    spectral_batch_size: int = 512    # fixed batch (see training/loop.py) the Hessian loss is computed on
    spectral_growth: float = 1.15     # geometric growth factor for the coarse logging phase
    spectral_gap_rate_threshold: float = 1e-4  # |d(train_acc - test_acc)/d(step)| that triggers dense logging
    spectral_dense_stride: int = 100  # step stride while densely logging through a transition

    # Optimization
    optimization_steps: int = 200_000
    optimizer: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 0.01
    loss_function: str = "CrossEntropy"

    # Adversarial eval
    fgsm_epsilon: float = 0.1
    fgsm_refresh_step: int = 20_000

    # Logging/saving
    log_every: int = 150
    eval_every: int = 1000
    checkpoint_every: int = 1000  # independent of eval_every -- see training/loop.py
    output_dir: str = os.environ.get("OUTPUT_DIR", "./runs_arch_sweep")
    seed: int = 0
    dtype: torch.dtype = torch.float32