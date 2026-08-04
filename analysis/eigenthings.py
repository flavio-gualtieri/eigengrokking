from __future__ import annotations

import math
import torch

import torch.nn as nn

from dataclasses import dataclass

from collections.abc import Iterable


class Eigenthings:
    def __init__(
              self,
              model: nn.Module,
              m: int,
              sigma: float,
              params: list[torch.Tensor],
              device: torch.device,
              loss,
              orthogonalize: bool = True,
    ):
        self.model = model
        self.m = m
        self.sigma = sigma
        self.loss = loss

        self.device = device
        self.params = params
        self.orthogonalize = orthogonalize

        # First-order grads w.r.t. a create_graph=True loss are what makes
        # HVPs possible (Hv = d/dparams (grad . v)), and the loss is fixed
        # for this snapshot -- so compute them once here rather than
        # re-walking the whole backward pass on every hvp() call.
        self.grads = torch.autograd.grad(self.loss, self.params, create_graph=True, allow_unused=True)

    def gaussian_kernel(self, t, center, sigma):
        coeff = sigma * math.sqrt(2.0 * math.pi)
        return torch.exp(-0.5 * ((t - center) / sigma) ** 2) / coeff

    def unpack_vec(self, v: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        i = 0
        for p in self.params:
            n = p.numel()
            out.append(v[i : i + n].view_as(p))
            i += n

        return out

    def pack_list(self, xs: Iterable[torch.Tensor]) -> torch.Tensor:
        xs = list(xs)
        if len(xs) == 0:
            return torch.empty(0, device=self.device)
        
        return torch.cat([x.reshape(-1) for x in xs], dim=0)

    def hvp(self, q):
        q = q.to(self.device)
        q_list = self.unpack_vec(q)

        gv = torch.zeros((), device=self.device, dtype=torch.float32)
        for g, v in zip(self.grads, q_list):
            if g is not None:
                gv = gv + (g * v).sum()

        Hv = torch.autograd.grad(gv, self.params, retain_graph=True, create_graph=False, allow_unused=True)

        # Handle None gradients by converting to zeros
        Hv_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(Hv, self.params)]
        return self.pack_list(Hv_list)

    def lanczos(self, v0):
        v0 = torch.as_tensor(v0, dtype=torch.float32, device=self.device)
        n = v0.shape[0]

        q_prev = torch.zeros(n, dtype=torch.float32, device=self.device)
        q = v0 / torch.linalg.norm(v0)

        alpha = []
        beta = []
        Q = [q.clone()]

        beta_prev = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for _ in range(self.m):
            w = self.hvp(q) - beta_prev * q_prev
            a = torch.dot(q, w)
            w = w - a * q

            if self.orthogonalize:
                for qi in Q:
                    w = w - torch.dot(qi, w) * qi

            b = torch.linalg.norm(w)

            alpha.append(a)

            if b < 1e-12:
                break

            beta.append(b)
            q_prev = q
            q = w / b
            Q.append(q.clone())
            beta_prev = b

        k = len(alpha)
        T = torch.zeros((k, k), dtype=torch.float32, device=self.device)
        for i in range(k):
            T[i, i] = alpha[i]
            if i < k - 1:
                T[i, i + 1] = beta[i]
                T[i + 1, i] = beta[i]

        Q = torch.stack(Q[:k], dim=1)

        alpha = torch.stack(alpha)
        beta = torch.stack(beta) if beta else torch.empty(0, dtype=torch.float32, device=self.device)

        return T

    def lanczos_quadrature(self, v0):
        T = self.lanczos(v0)

        T_cpu = T.to("cpu")
        eigvals, eigvecs = torch.linalg.eigh(T_cpu)

        nodes = eigvals.to(self.device)
        weights = (eigvecs[0, :] ** 2).to(self.device)

        return nodes, weights

    def density_from_lanczos(self, nodes, weights, t_grid, sigma):
        density = torch.zeros_like(t_grid)
        for lmbda, w in zip(nodes, weights):
            if w.abs() > 1e-14:  # Skip negligibly small weights
                density += w * self.gaussian_kernel(t_grid, lmbda, sigma)
    
        return density


def rademacher_probe(
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
) -> torch.Tensor:
    """
    Draws a +-1 Rademacher probe vector. Sampled with a CPU generator (CUDA/MPS
    generators aren't interchangeable, so this keeps probe sequences identical
    regardless of accelerator) and moved to `device` afterwards.
    """
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64)
    return (signs.to(dtype) * 2 - 1).to(device)


@dataclass
class SpectralEstimate:
    """
    Result of one SLQ spectral density estimate: the probe-averaged smoothed
    density (for plotting) plus the raw per-probe Ritz nodes/weights (for
    scalar observables and probe-wise error bars -- see
    analysis/spectral_observables.py).
    """
    density: torch.Tensor
    t_grid: torch.Tensor
    probe_nodes: list[torch.Tensor]
    probe_weights: list[torch.Tensor]
    n_params: int
    sigma: float  # realized absolute KDE bandwidth = sigma_frac * (cheap pre-pass top_eig estimate)


def estimate_density(
        model,
        m,
        k,
        sigma_frac,
        loss,
        probe_seed: int,
        top_eig_probe_m: int = 20,
        eps: float = 1e-12,
    ) -> SpectralEstimate:
    """
    Stochastic Lanczos Quadrature estimate of the Hessian's spectral density.

    `probe_seed` is reseeded into a fresh generator on every call, so calling
    this repeatedly with the same `probe_seed` (e.g. once per checkpoint in a
    training run) draws bit-identical probe vectors every time -- isolating
    parameter-driven spectral change from probe-sampling noise across
    checkpoints.

    `sigma_frac` sets the KDE bandwidth as a fraction of lambda_max rather
    than an absolute value. lambda_max can move by orders of magnitude over
    a training run, so a fixed absolute sigma makes the bandwidth -- and
    therefore how comparable the plotted density is -- inconsistent between
    checkpoints, and it interacts badly with the `nt` cap below: dt=sigma/5
    needs to shrink alongside the node range (which scales with lambda_max)
    or nt=(hi-lo)/dt can blow past the 50000 cap and get silently coarsened
    past sigma/5. Getting an absolute sigma requires knowing lambda_max
    first, so a cheap single-probe, m=`top_eig_probe_m` Lanczos pass
    estimates it before the real k-probe, m-step pass runs -- top_eig
    converges fast with m (see reports/spectral_validation.md), so a shallow
    pre-pass is enough. The realized absolute sigma is returned on
    `SpectralEstimate.sigma` since callers no longer choose it directly.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    device = next(model.parameters()).device

    n = sum(p.numel() for p in params)

    eigen = Eigenthings(
        model=model,
        m=top_eig_probe_m,
        sigma=0.0,
        params=params,
        device=device,
        loss=loss,
        )

    # Own generator, independent of the main k-probe draws below, so adding
    # this pre-pass doesn't perturb the probe sequence everything else uses.
    pre_generator = torch.Generator().manual_seed(probe_seed)
    v0 = rademacher_probe(n, device, dtype=torch.float32, generator=pre_generator)
    v0 = v0 / torch.linalg.norm(v0)
    pre_nodes, _ = eigen.lanczos_quadrature(v0)
    top_eig_estimate = abs(pre_nodes.max().item())
    sigma = max(sigma_frac * top_eig_estimate, eps)

    # Reuse the same instance (same Eigenthings.grads, computed once in
    # __init__) for the real pass rather than constructing a second one.
    eigen.m = m
    eigen.sigma = sigma

    generator = torch.Generator().manual_seed(probe_seed)

    all_nodes = []
    all_weights = []

    for _ in range(k):
        v = rademacher_probe(n, device, dtype=torch.float32, generator=generator)
        v = v / torch.linalg.norm(v)
        nodes, weights = eigen.lanczos_quadrature(v)
        all_nodes.append(nodes)
        all_weights.append(weights)

    nodes_cat = torch.cat(all_nodes)
    lo = nodes_cat.min().item() - 6 * sigma
    hi = nodes_cat.max().item() + 6 * sigma

    dt = sigma / 5
    nt = int((hi - lo) / dt) + 1
    nt = max(200, min(50000, nt))

    t_grid = torch.linspace(lo, hi, nt, device=device, dtype=torch.float32)
    density = torch.zeros_like(t_grid)

    for nodes, weights in zip(all_nodes, all_weights):
        density += eigen.density_from_lanczos(nodes, weights, t_grid, sigma)

    density /= k

    return SpectralEstimate(
        density=density,
        t_grid=t_grid,
        probe_nodes=all_nodes,
        probe_weights=all_weights,
        n_params=n,
        sigma=sigma,
    )