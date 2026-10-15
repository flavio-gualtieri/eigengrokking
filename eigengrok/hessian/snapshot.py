# eigengrok/hessian/snapshot.py

"""
One Hessian snapshot: SLQ statistics of a model's training loss on the fixed
evaluation batch. Shared by the training loop (live snapshots) and
scripts/spectral_from_checkpoints.py (post-hoc, from checkpoints).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from eigengrok.hessian.operators import CurvatureOperators, LossFn
from eigengrok.hessian.spectrum import slq, spectral_density, spectral_stats_with_stderr


def fixed_eval_batch(
        dataset: torch.utils.data.Dataset,
        max_points: int,
        seed: int,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The batch every Hessian snapshot of a run is computed on: the whole
    training set if it has at most `max_points` examples, otherwise a subset
    drawn with `seed`. Fixed for the whole run, so snapshots differ only
    through the parameters, never through which minibatch came last.
    """
    n = len(dataset)
    if n <= max_points:
        indices = list(range(n))
    else:
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        indices = sorted(perm[:max_points].tolist())
    subset = torch.utils.data.Subset(dataset, indices)
    x, y = next(iter(torch.utils.data.DataLoader(subset, batch_size=len(indices), shuffle=False)))
    # Same input cast as the training step (eigengrok/train/loop.py).
    return x.to(device).float(), y.to(device)


@dataclass
class HessianSnapshot:
    stats: dict[str, float]
    stderr: dict[str, float]
    grid: np.ndarray     # plotting only
    density: np.ndarray  # plotting only


OPERATORS = {"hessian": "hvp", "gauss_newton": "gauss_newton", "functional": "functional"}


def operator_stats(
        ops: CurvatureOperators,
        operator: str,
        *,
        m: int,
        k: int,
        probe_seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """SLQ statistics (point estimate, stderr over probes) of one curvature operator of `ops`."""
    matvec = getattr(ops, OPERATORS[operator])
    q = slq(matvec, ops.n_params, k=k, m=m, seed=probe_seed, device=ops.theta.device, dtype=ops.theta.dtype)
    return spectral_stats_with_stderr(q)


def hessian_snapshot(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: LossFn,
        *,
        m: int,
        k: int,
        probe_seed: int,
        sigma_frac: float = 0.01,
) -> HessianSnapshot:
    ops = CurvatureOperators(model, x, y, loss_fn)
    q = slq(ops.hvp, ops.n_params, k=k, m=m, seed=probe_seed, device=ops.theta.device, dtype=ops.theta.dtype)
    stats, stderr = spectral_stats_with_stderr(q)
    grid, density = spectral_density(q, sigma_frac=sigma_frac)
    return HessianSnapshot(stats=stats, stderr=stderr, grid=grid, density=density)
