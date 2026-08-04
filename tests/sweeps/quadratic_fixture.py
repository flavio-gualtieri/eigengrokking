# tests/sweeps/quadratic_fixture.py

"""
Engineered quadratic-form fixture with a chosen, production-like spectral
dynamic range -- see reports/spectral_validation.md ("Extended fixture:
production dynamic range").

The original tiny_mlp fixture (tests/test_eigenthings.py, exact spectrum via
tests/reference_hessian.py) is random-init, so its spectrum is one roughly
flat bulk -- it never exercises what bulk_edge/conditioning measure: how far
an outlier eigenvalue sits above the bulk. Rather than training a net (extra
hyperparameters to justify) to reach that regime, this fixture chooses the
exact spectrum directly: analysis/eigenthings.py's hvp() differentiates a
`loss` tensor handed in fully-formed -- estimate_density() never calls
model.forward() -- so `loss` doesn't need to come from a neural net. A
quadratic form `loss = 0.5 * theta^T M theta` (`theta` a bare nn.Parameter)
has Hessian w.r.t. `theta` equal to `M` *exactly*, so choosing `M`'s
eigenvalues chooses the exact spectrum with zero materialization cost -- no
HVP-based column-by-column Hessian needed at all.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from analysis.spectral_observables import SpectralObservables, compute_spectral_observables

N_PARAMS = 1026  # matches tests/test_eigenthings.py's tiny_mlp param count, for comparability
BULK_RANGE = (-0.3, 0.3)
OUTLIERS = (40.0, 60.0, 80.0, 100.0, 150.0)
SEED = 0


class QuadraticForm(nn.Module):
    """loss = 0.5 * theta^T M theta -> Hessian w.r.t. theta is exactly M."""

    def __init__(self, M: torch.Tensor):
        super().__init__()
        self.register_buffer("M", M)
        self.theta = nn.Parameter(torch.zeros(M.shape[0]))

    def energy(self) -> torch.Tensor:
        return 0.5 * self.theta @ (self.M @ self.theta)


def build_M(n: int, eigvals: np.ndarray, seed: int) -> torch.Tensor:
    """M = Q diag(eigvals) Q^T, Q orthogonal via QR of a fixed-seed Gaussian."""
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(n, n, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(G)
    lam = torch.as_tensor(eigvals, dtype=torch.float64)
    M = Q @ torch.diag(lam) @ Q.T
    M = 0.5 * (M + M.T)  # kill float asymmetry
    return M.to(torch.float32)


def make_fixture(
        n: int = N_PARAMS,
        outliers: tuple[float, ...] = OUTLIERS,
        bulk_range: tuple[float, float] = BULK_RANGE,
        seed: int = SEED,
    ) -> tuple[QuadraticForm, torch.Tensor, torch.Tensor]:
    """Returns (model, loss, exact_spectrum) -- exact_spectrum sorted ascending."""
    rng = np.random.default_rng(seed)
    n_bulk = n - len(outliers)
    bulk = rng.uniform(bulk_range[0], bulk_range[1], size=n_bulk)
    eigvals = np.concatenate([bulk, np.array(outliers)])
    rng.shuffle(eigvals)

    M = build_M(n, eigvals, seed=seed)
    model = QuadraticForm(M)
    loss = model.energy()

    exact_spectrum = torch.as_tensor(np.sort(eigvals), dtype=torch.float32)
    return model, loss, exact_spectrum


def exact_observables(exact_spectrum: torch.Tensor) -> SpectralObservables:
    """
    Exact reference for all SpectralObservables fields: feed the full exact
    spectrum into compute_spectral_observables as a single pseudo-probe with
    uniform weight 1/n_params -- same trick tests/test_eigenthings.py uses
    for the tiny_mlp fixture, so the approximate-vs-exact comparison reuses
    the real bulk_edge/entropy logic instead of re-deriving it by hand.
    """
    n = exact_spectrum.numel()
    return compute_spectral_observables(
        probe_nodes=[exact_spectrum],
        probe_weights=[torch.full_like(exact_spectrum, 1.0 / n)],
        n_params=n,
    )
