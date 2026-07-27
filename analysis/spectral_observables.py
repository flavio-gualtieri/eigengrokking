# analysis/spectral_observables.py

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import List

import numpy as np
import torch


@dataclass
class SpectralObservables:
    top_eig: float          # lambda_max: largest Ritz value across all probes
    bulk_edge: float        # median + bulk_mad_multiplier * MAD(nodes): robust bulk/outlier threshold
    outlier_count: int      # distinct node clusters beyond bulk_edge
    trace: float            # Hutchinson-via-Lanczos estimate of tr(H)
    spectral_entropy: float # Shannon entropy of the |lambda| spectral measure
    effective_rank: float   # exp(spectral_entropy)
    negative_mass: float    # probability mass at lambda < 0
    conditioning: float     # top_eig / bulk_edge, a conditioning proxy

    def as_dict(self) -> dict:
        return asdict(self)


def _pool_probes(
        probe_nodes: List[torch.Tensor],
        probe_weights: List[torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Pools per-probe Ritz (node, weight) pairs into one discrete distribution
    approximating the normalized spectral measure (1/n) sum_i delta(t - lambda_i):
    each of the k probes contributes total mass 1/k.
    """
    k = len(probe_nodes)
    nodes = np.concatenate([t.detach().cpu().numpy() for t in probe_nodes])
    weights = np.concatenate([t.detach().cpu().numpy() for t in probe_weights]) / k
    return nodes, weights


def compute_spectral_observables(
        probe_nodes: List[torch.Tensor],
        probe_weights: List[torch.Tensor],
        n_params: int,
        bulk_mad_multiplier: float = 5.0,
        resolution_frac: float = 0.05,
        eps: float = 1e-12,
    ) -> SpectralObservables:
    """
    Computes checkpoint-level scalar summaries of the Hessian spectrum from
    the *raw* (pre-KDE) per-probe SLQ output. Working from the raw pooled
    (node, weight) atoms -- rather than the KDE-smoothed density used for
    plotting -- avoids extra smoothing-bandwidth bias in tail-sensitive
    quantities like top_eig and outlier detection.

    `resolution_frac` sets one shared length scale (as a fraction of
    |top_eig|) used both to decide when two outlier nodes are "the same"
    eigenvalue rediscovered by different probes, and as the bin width for the
    entropy histogram.
    """
    nodes, weights = _pool_probes(probe_nodes, probe_weights)
    if nodes.size == 0:
        raise ValueError("No Lanczos nodes available; check that Lanczos did not degenerate.")

    order = np.argsort(nodes)
    nodes, weights = nodes[order], weights[order]

    # --- trace (Hutchinson-via-Lanczos) ---
    trace = float(n_params * np.sum(weights * nodes))

    # --- top_eig ---
    top_eig = float(nodes[-1])
    resolution = max(resolution_frac * abs(top_eig), eps)

    # --- bulk edge: robust (median + c*MAD) threshold on the pooled nodes ---
    median = float(np.median(nodes))
    mad = float(np.median(np.abs(nodes - median))) * 1.4826  # normal-consistent scale
    bulk_edge = median + bulk_mad_multiplier * max(mad, eps)

    # --- outlier count: distinct clusters of nodes beyond the bulk edge ---
    outlier_nodes = nodes[nodes > bulk_edge]
    outlier_count = 0
    if outlier_nodes.size > 0:
        gaps = np.diff(outlier_nodes) > resolution
        outlier_count = int(gaps.sum() + 1)

    # --- negative-eigenvalue mass (exact, from pooled weights) ---
    negative_mass = float(weights[nodes < 0].sum())

    # --- spectral entropy / effective rank, over |lambda| ---
    abs_nodes = np.abs(nodes)
    abs_max = max(float(abs_nodes.max()), eps)
    n_bins = max(int(np.ceil(abs_max / resolution)), 1)
    hist, _ = np.histogram(abs_nodes, bins=n_bins, range=(0.0, abs_max), weights=weights)
    p = hist / max(hist.sum(), eps)
    p = p[p > eps]
    spectral_entropy = float(-np.sum(p * np.log(p)))
    effective_rank = float(math.exp(spectral_entropy))

    conditioning = float(top_eig / bulk_edge) if abs(bulk_edge) > eps else float("nan")

    return SpectralObservables(
        top_eig=top_eig,
        bulk_edge=bulk_edge,
        outlier_count=outlier_count,
        trace=trace,
        spectral_entropy=spectral_entropy,
        effective_rank=effective_rank,
        negative_mass=negative_mass,
        conditioning=conditioning,
    )


def compute_spectral_observables_with_stderr(
        probe_nodes: List[torch.Tensor],
        probe_weights: List[torch.Tensor],
        n_params: int,
        **kwargs,
    ) -> tuple[SpectralObservables, SpectralObservables]:
    """
    Returns (point_estimate, stderr).

    `point_estimate` pools all k probes together (the least-biased estimate
    of each quantity). `stderr` is the standard error of the mean across the
    k single-probe estimates of the same quantity -- i.e. "error bars over
    probes": how much each number would move under a fresh draw of probes.

    Note this isn't the literal SEM of `point_estimate` for every field (e.g.
    top_eig is a pooled max, not a mean, so it's not equal to the mean of the
    per-probe maxes) -- it's the standard, pragmatic proxy for probe-sampling
    uncertainty used throughout.
    """
    point = compute_spectral_observables(probe_nodes, probe_weights, n_params, **kwargs)

    k = len(probe_nodes)
    per_probe = [
        compute_spectral_observables([probe_nodes[i]], [probe_weights[i]], n_params, **kwargs)
        for i in range(k)
    ]

    stderr_values = {}
    for f in fields(point):
        vals = np.array([getattr(p, f.name) for p in per_probe], dtype=float)
        vals = vals[np.isfinite(vals)]
        stderr_values[f.name] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")

    stderr = SpectralObservables(**stderr_values)
    return point, stderr
