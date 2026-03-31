from __future__ import annotations

from configs.configs import ExperimentConfig

EXPERIMENTS = [
    ExperimentConfig(
        dataset='MNIST',
        optimization_steps=5_000,
        initialization_scale=8,
        depth=3,
        width=200,
        test_mode=True,
    )
]
