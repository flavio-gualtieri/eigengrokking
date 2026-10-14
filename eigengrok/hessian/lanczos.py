# eigengrok/hessian/lanczos.py

"""
Batched Lanczos with full reorthogonalisation.

`lanczos` runs k independent Lanczos recurrences side by side, one per start
vector, so each iteration costs a single batched matvec. Each new vector is
orthogonalised against the whole basis built so far, twice (classical
Gram-Schmidt run twice is numerically as good as modified Gram-Schmidt).
Without that, Lanczos loses orthogonality once an extreme eigenvalue converges
and "ghost" copies of it reappear.

For a start vector q with ||q|| = 1, the eigendecomposition T = S diag(theta) S^T
of the m x m tridiagonal gives the m-point Gauss quadrature rule of the
spectral measure mu_q = sum_i (q . u_i)^2 delta(lambda_i): nodes theta_j,
weights S[0, j]^2. It integrates polynomials of degree <= 2m - 1 exactly, i.e.
sum_j w_j theta_j^n = q^T H^n q for n <= 2m - 1, with no grid or smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

MatVec = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class LanczosResult:
    alpha: torch.Tensor          # [k, m] diagonals of T (float64, CPU)
    beta: torch.Tensor           # [k, m] off-diagonals; beta[:, j] couples steps j and j+1
    steps: torch.Tensor          # [k] Krylov dimension built per probe (< m after an early breakdown)
    basis: torch.Tensor | None   # [k, m, P] Lanczos vectors, only if requested

    def tridiagonal(self, i: int) -> np.ndarray:
        n = int(self.steps[i])
        a = self.alpha[i, :n].numpy()
        b = self.beta[i, : n - 1].numpy()
        return np.diag(a) + np.diag(b, 1) + np.diag(b, -1)

    def quadrature(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per-probe Gauss quadrature nodes and weights (each probe's weights sum to 1)."""
        nodes, weights = [], []
        for i in range(len(self.steps)):
            theta, S = np.linalg.eigh(self.tridiagonal(i))
            nodes.append(theta)
            weights.append(S[0] ** 2)
        return nodes, weights


def lanczos(
        matvec: MatVec,
        start: torch.Tensor,
        m: int,
        *,
        keep_basis: bool = False,
) -> LanczosResult:
    """
    m steps of Lanczos from each row of `start` ([k, P]), on the symmetric
    operator `matvec` ([k, P] -> [k, P]). A probe whose Krylov space is
    exhausted (an invariant subspace was found, e.g. m >= P) stops early; its
    `steps` records how far it got and its later rows are zero.
    """
    k, n = start.shape
    dtype, device = start.dtype, start.device
    # A probe has exhausted its Krylov space when the new residual is at
    # round-off level relative to the largest |alpha| + beta seen so far.
    # Kept tight (float32: 1.2e-5): trained-net spectra span many decades,
    # so a looser threshold would stop early in the bulk, a false breakdown.
    breakdown_tol = 100 * torch.finfo(dtype).eps

    basis = torch.zeros(k, m, n, dtype=dtype, device=device)
    alpha = torch.zeros(k, m, dtype=torch.float64)
    beta = torch.zeros(k, m, dtype=torch.float64)
    steps = torch.zeros(k, dtype=torch.long)
    active = torch.ones(k, dtype=torch.bool, device=device)
    scale = torch.zeros(k, dtype=dtype, device=device)  # running max of |alpha| + beta, for a relative breakdown test

    q = start / start.norm(dim=1, keepdim=True)
    beta_prev = torch.zeros(k, dtype=dtype, device=device)

    for j in range(m):
        basis[:, j] = q
        w = matvec(q)
        a = (q * w).sum(dim=1)
        w = w - a[:, None] * q
        if j > 0:
            w = w - beta_prev[:, None] * basis[:, j - 1]
        for _ in range(2):
            Q = basis[:, : j + 1]
            w = w - torch.einsum("kjp,kj->kp", Q, torch.einsum("kjp,kp->kj", Q, w))
        b = w.norm(dim=1)

        a = torch.where(active, a, torch.zeros_like(a))
        alpha[:, j] = a.double().cpu()
        steps += active.cpu().long()

        scale = torch.maximum(scale, a.abs() + b)
        active = active & (b > breakdown_tol * scale)
        b = torch.where(active, b, torch.zeros_like(b))
        beta[:, j] = b.double().cpu()

        if not bool(active.any()):
            break
        q = torch.where(active[:, None], w / b.clamp_min(torch.finfo(dtype).tiny)[:, None], torch.zeros_like(w))
        beta_prev = b

    return LanczosResult(alpha=alpha, beta=beta, steps=steps, basis=basis if keep_basis else None)


def ritz_pairs(result: LanczosResult, probe: int = 0) -> tuple[np.ndarray, torch.Tensor]:
    """
    Ritz values (ascending) and Ritz vectors ([n, P], unit norm) from one
    probe's Krylov basis. Requires `lanczos(..., keep_basis=True)`.
    """
    if result.basis is None:
        raise ValueError("Ritz vectors need the Lanczos basis: call lanczos(..., keep_basis=True)")
    n = int(result.steps[probe])
    theta, S = np.linalg.eigh(result.tridiagonal(probe))
    Q = result.basis[probe, :n]
    vectors = torch.as_tensor(S.T, dtype=Q.dtype, device=Q.device) @ Q
    return theta, vectors / vectors.norm(dim=1, keepdim=True)
