# eigengrok/hessian/spectrum.py

"""
Hessian spectral statistics by stochastic Lanczos quadrature (SLQ).

k Rademacher probes each give an m-point Gauss quadrature rule (nodes theta,
weights w summing to 1) for the spectral measure the probe sees
(eigengrok/hessian/lanczos.py). Averaging the k rules estimates the
normalised spectral measure mu = (1/P) sum_i delta(lambda_i) of the P x P
Hessian, and every scalar statistic below is an integral against that
discrete measure. Nothing goes through a grid: polynomial moments up to
degree 2m - 1 are exact per probe, so their only error is the Monte Carlo
error of the probe average. The Gaussian-smoothed density is for plotting only.

Statistics (spectral_stats):
    top_eig, min_eig       largest / smallest node over all probes
    trace                  P * int lambda dmu  (Hutchinson's estimator with these probes)
    moment_n               int lambda^n dmu ~ tr(H^n) / P, for n = 1..4
    normalized_moment_n    tr(H^n) / tr(H)^n, for n = 2..4 (moments of H / tr(H); scale-invariant)
    top_over_trace         top_eig / trace (scale-invariant)
    participation_ratio    tr(H)^2 / tr(H^2) (scale-invariant)
    stable_rank            tr(H^2) / top_eig^2 (scale-invariant)
    spectral_entropy       entropy of the |lambda|-weighted measure; effective_rank = exp of it
    negative_mass          mu(lambda < -negative_margin * |top_eig|)
    n_outliers             P * mu(lambda > outlier_frac * top_eig): eigenvalues within
                           1/outlier_frac of the top one, counted with multiplicity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from eigengrok.hessian.lanczos import lanczos, ritz_pairs

MatVec = Callable[[torch.Tensor], torch.Tensor]

MOMENT_ORDERS = (1, 2, 3, 4)

# Every key spectral_stats returns, in order.
STAT_NAMES = (
    "top_eig", "min_eig", "trace",
    *(f"moment_{n}" for n in MOMENT_ORDERS),
    *(f"normalized_moment_{n}" for n in MOMENT_ORDERS if n > 1),
    "top_over_trace", "participation_ratio", "stable_rank",
    "spectral_entropy", "effective_rank", "negative_mass", "n_outliers",
)


def rademacher_probes(k: int, n: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    k Rademacher (+-1) vectors of length n, shape [k, n]. Drawn from a fresh
    CPU generator seeded with `seed`, so the same seed gives the same probes
    at every snapshot of a run and on any device.
    """
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (k, n), generator=generator, dtype=torch.int64)
    return (signs.to(dtype) * 2 - 1).to(device)


@dataclass
class SpectralQuadrature:
    nodes: list[np.ndarray]    # per probe, float64
    weights: list[np.ndarray]  # per probe, each sums to 1
    n_params: int

    @property
    def k(self) -> int:
        return len(self.nodes)

    def pooled(self) -> tuple[np.ndarray, np.ndarray]:
        """All probes' atoms as one measure, each probe carrying total mass 1/k."""
        return np.concatenate(self.nodes), np.concatenate(self.weights) / self.k

    def probe(self, i: int) -> SpectralQuadrature:
        return SpectralQuadrature([self.nodes[i]], [self.weights[i]], self.n_params)


def exact_quadrature(eigenvalues) -> SpectralQuadrature:
    """
    The exact spectral measure of a materialised matrix, as a single "probe"
    with weight 1/P on every eigenvalue, so spectral_stats of it gives the
    exact value of every statistic (the reference SLQ is validated against).
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    return SpectralQuadrature([lam], [np.full(lam.size, 1.0 / lam.size)], lam.size)


def slq(matvec: MatVec, n_params: int, *, k: int, m: int, seed: int,
        device: torch.device, dtype: torch.dtype, probe_batch: int | None = None) -> SpectralQuadrature:
    """
    Runs k-probe, m-step SLQ on the symmetric operator `matvec` ([b, P] -> [b, P]).
    Probes are independent, so `probe_batch` runs them in chunks of that size
    (same result, smaller Lanczos basis in memory); None runs all k together.
    """
    probes = rademacher_probes(k, n_params, seed=seed, device=device, dtype=dtype)
    nodes, weights = [], []
    for chunk in probes.split(probe_batch or k):
        chunk_nodes, chunk_weights = lanczos(matvec, chunk, m).quadrature()
        nodes += chunk_nodes
        weights += chunk_weights
    return SpectralQuadrature(nodes=nodes, weights=weights, n_params=n_params)


def spectral_stats(
        q: SpectralQuadrature,
        *,
        negative_margin: float = 0.05,
        outlier_frac: float = 0.1,
        eps: float = 1e-12,
) -> dict[str, float]:
    nodes, weights = q.pooled()
    P = q.n_params

    moments = {n: float(np.sum(weights * nodes ** n)) for n in MOMENT_ORDERS}
    trace = P * moments[1]
    trace_sq = P * moments[2]
    top = float(nodes.max())

    stats = {
        "top_eig": top,
        "min_eig": float(nodes.min()),
        "trace": trace,
        **{f"moment_{n}": moments[n] for n in MOMENT_ORDERS},
        **{f"normalized_moment_{n}": _safe_div(P * moments[n], trace ** n) for n in MOMENT_ORDERS if n > 1},
        "top_over_trace": _safe_div(top, trace),
        "participation_ratio": _safe_div(trace ** 2, trace_sq),
        "stable_rank": _safe_div(trace_sq, top ** 2),
    }

    # Entropy of the |lambda|-weighted measure, in closed form over the atoms:
    # treating each of the P eigenvalues as carrying ~1/P of the mass,
    #   S = log(P * sum_j w_j|theta_j|) - sum_j w_j|theta_j| log|theta_j| / sum_j w_j|theta_j|.
    # P (not the number of atoms, which grows with m*k) keeps S independent of the Lanczos settings.
    abs_nodes = np.abs(nodes)
    weighted_abs = weights * abs_nodes
    z = max(float(weighted_abs.sum()), eps)
    cross = float(np.sum(weighted_abs * np.log(np.maximum(abs_nodes, eps))))
    stats["spectral_entropy"] = math.log(P * z) - cross / z
    stats["effective_rank"] = math.exp(stats["spectral_entropy"])

    # Thresholds are relative to |top_eig|: Lanczos puts small spurious nodes
    # either side of a zero eigenvalue, which a raw `< 0` would count.
    stats["negative_mass"] = float(weights[nodes < -negative_margin * abs(top)].sum())
    stats["n_outliers"] = float(P * weights[nodes > outlier_frac * top].sum()) if top > 0 else 0.0
    return stats


def spectral_stats_with_stderr(q: SpectralQuadrature, **kwargs) -> tuple[dict[str, float], dict[str, float]]:
    """
    (point estimate pooled over all probes, standard error over probes). The
    stderr is the SEM of the k single-probe estimates of each statistic, i.e.
    how much it would move under a fresh draw of probes. For the extremes
    (top_eig, min_eig) and ratios this is a proxy, not the literal SEM.
    """
    point = spectral_stats(q, **kwargs)
    per_probe = [spectral_stats(q.probe(i), **kwargs) for i in range(q.k)]
    stderr = {}
    for key in point:
        vals = np.array([s[key] for s in per_probe], dtype=float)
        vals = vals[np.isfinite(vals)]
        stderr[key] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    return point, stderr


def spectral_density(q: SpectralQuadrature, *, sigma_frac: float = 0.01,
                     n_grid: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-smoothed spectral density on a grid, for plotting only. The
    bandwidth is `sigma_frac` of the node range and the grid always has
    `n_grid` points, so spacing stays far below the bandwidth however wide
    the spectrum is.
    """
    nodes, weights = q.pooled()
    lo, hi = float(nodes.min()), float(nodes.max())
    sigma = sigma_frac * (hi - lo) if hi > lo else max(abs(hi), 1.0) * sigma_frac
    grid = np.linspace(lo - 6 * sigma, hi + 6 * sigma, n_grid)
    density = np.zeros_like(grid)
    for node, w in zip(nodes, weights):
        density += w * np.exp(-0.5 * ((grid - node) / sigma) ** 2)
    return grid, density / (sigma * math.sqrt(2 * math.pi))


def hutchinson_trace(matvec: MatVec, n_params: int, *, k: int, seed: int,
                     device: torch.device, dtype: torch.dtype) -> tuple[float, float]:
    """tr(A) ~ mean_i v_i^T A v_i over k Rademacher probes: (estimate, standard error)."""
    V = rademacher_probes(k, n_params, seed=seed, device=device, dtype=dtype)
    samples = (V * matvec(V)).sum(dim=1).double().cpu().numpy()
    stderr = float(samples.std(ddof=1) / math.sqrt(k)) if k > 1 else float("nan")
    return float(samples.mean()), stderr


def top_eigenpairs(matvec: MatVec, n_params: int, *, n_top: int, m: int, seed: int,
                   device: torch.device, dtype: torch.dtype) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """
    The n_top largest Ritz pairs from one m-step Lanczos run: eigenvalues
    (descending), unit eigenvectors [n_top, P], and residual norms
    ||A v - theta v||, which say how converged each pair actually is.
    """
    start = rademacher_probes(1, n_params, seed=seed, device=device, dtype=dtype)
    theta, vectors = ritz_pairs(lanczos(matvec, start, m, keep_basis=True))
    order = np.argsort(theta)[::-1][:n_top]
    values = theta[order].copy()
    vecs = vectors[torch.as_tensor(order.copy(), device=vectors.device)]
    residual = matvec(vecs) - torch.as_tensor(values, dtype=vecs.dtype, device=vecs.device)[:, None] * vecs
    return values, vecs, residual.norm(dim=1).double().cpu().numpy()


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")
