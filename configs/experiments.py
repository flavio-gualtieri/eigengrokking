from __future__ import annotations

from configs.config import ExperimentConfig

EXPERIMENTS = [
    ExperimentConfig(
        dataset='MNIST',
        optimization_steps=10_000,
        train_points=1000,
        batch_size=64,
        weight_decay=0.01,
        initialization_scale=8,
        run_spectral=True,
        test_mode=True,
    ),
]
