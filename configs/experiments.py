from __future__ import annotations

from configs.base import ExperimentConfig
from configs.sweeps import with_seeds

_MODULUS = 97
_TRAIN_FRAC = 0.3
_FULL_BATCH = 8192  # > modulus**2 * train_frac (~2823), so training is effectively full-batch
_SEEDS = range(5)   # >=5 seeds so grokking/spectral curves get error bars, not single-seed noise

_MODULAR_ADD_DEFAULTS = dict(
    task="MODULAR",
    modulus=_MODULUS,
    train_frac=_TRAIN_FRAC,
    output_dim=_MODULUS,
    num_heads=4,
    activation="GELU",
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=1.0,
    loss_function="CrossEntropy",
    batch_size=_FULL_BATCH,
    optimization_steps=30_000,
    initialization_scale=1.0,
    log_every=100,
    eval_every=100,
    checkpoint_every=100,  # matches eval_every -- keeps checkpoint cadence unchanged now that it's decoupled
)

_BASE_EXPERIMENTS = [
    ExperimentConfig(
        **_MODULAR_ADD_DEFAULTS,
        depth=1,
        width=128,
    ),
    ExperimentConfig(
        **_MODULAR_ADD_DEFAULTS,
        depth=2,
        width=128,
    ),
]

EXPERIMENTS = with_seeds(_BASE_EXPERIMENTS, seeds=_SEEDS)
