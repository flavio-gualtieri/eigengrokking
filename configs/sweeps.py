# configs/sweeps.py

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

from configs.base import ExperimentConfig


def with_seeds(base_cfgs: Iterable[ExperimentConfig], seeds: Iterable[int]) -> List[ExperimentConfig]:
    """
    Expands each base config into one copy per seed. Single-seed grokking
    curves (and single-seed spectral statistics) are notoriously variable --
    this is how a run family gets enough seeds for error bars.
    """
    seeds = list(seeds)
    return [replace(cfg, seed=seed) for cfg in base_cfgs for seed in seeds]
