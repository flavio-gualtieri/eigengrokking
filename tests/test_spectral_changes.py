# tests/test_spectral_changes.py

"""
Shared probes (the same probe seed at every checkpoint of a run) measure the
change of a spectral statistic between two nearby parameter states far more
precisely than the statistic itself. Fast regression check on the exact MLP
fixture of tests/test_hessian.py; the full sweep over (m, k) is
scripts/validate_spectral_changes.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from eigengrok.hessian.operators import CurvatureOperators
from eigengrok.hessian.spectrum import exact_quadrature, slq, spectral_stats
from tests.test_hessian import CPU, F64, _materialise, _mlp_problem

SEEDS = range(12)


def _ops_at(ops: CurvatureOperators, theta: torch.Tensor) -> CurvatureOperators:
    vector_to_parameters(theta, ops.model.parameters())
    return CurvatureOperators(ops.model, ops.inputs, ops.targets, ops.loss_fn)


def test_shared_probes_measure_changes_precisely():
    ops0, _ = _mlp_problem("CrossEntropy")
    theta0 = parameters_to_vector(ops0.model.parameters()).detach().clone()
    delta = torch.randn(theta0.shape, generator=torch.Generator().manual_seed(1), dtype=F64)
    theta1 = theta0 + delta * (0.01 * theta0.norm() / delta.norm())

    states = [_ops_at(ops0, theta0), _ops_at(ops0, theta1)]
    exact = [spectral_stats(exact_quadrature(torch.linalg.eigvalsh(_materialise(o.hvp, o.n_params)).numpy()))
             for o in states]
    est = [[spectral_stats(slq(o.hvp, o.n_params, k=10, m=20, seed=s, device=CPU, dtype=F64)) for s in SEEDS]
           for o in states]

    for stat in ("trace", "effective_rank", "participation_ratio"):
        s0, s1 = exact[0][stat], exact[1][stat]
        a0 = np.array([e[stat] for e in est[0]])
        a1 = np.array([e[stat] for e in est[1]])
        level = np.median(np.abs(a0 - s0))
        same = np.median(np.abs((a1 - a0) - (s1 - s0)))
        indep = np.median(np.abs((a1 - np.roll(a0, 1)) - (s1 - s0)))
        assert same < 0.2 * level, (stat, same, level)
        assert same < 0.2 * indep, (stat, same, indep)
