# tests/test_eigenthings.py

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.mlp import MLP
from analysis.eigenthings import Eigenthings, estimate_density, rademacher_probe
from analysis.spectral_observables import compute_spectral_observables
from analysis.legacy.toy_eigenthings import compute_hessian_eigenvalues

DEVICE = torch.device("cpu")  # keeps test_determinism_fixed_seed meaningful


@pytest.fixture(scope="module")
def tiny_mlp():
    """
    Fixed-seed tiny MLP + one forward pass -> one loss tensor. Module-scoped:
    nothing below mutates model parameters, so every test in this file reads
    off the same loss graph (safe because every HVP/Hessian call in
    eigenthings.py and legacy/toy_eigenthings.py uses create_graph=True /
    retain_graph=True, so the graph never gets freed).
    """
    torch.manual_seed(0)
    model = MLP(
        input_dim=10,
        hidden_dims=(60,),
        output_dim=6,
        device=DEVICE,
        activation_fn="Tanh",  # smooth: avoids ReLU-kink edge cases at a fixed seed
        dropout=0.0,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    assert 800 <= n_params <= 2000, f"got {n_params} params, resize hidden_dims"

    x = torch.randn(16, 10, device=DEVICE)
    y = torch.randint(0, 6, (16,), device=DEVICE)
    loss = nn.functional.cross_entropy(model(x), y)

    return model, params, loss


@pytest.fixture(scope="module")
def exact_spectrum(tiny_mlp):
    model, _, loss = tiny_mlp
    return compute_hessian_eigenvalues(loss, model)  # sorted ascending, shape [n_params]


def test_trace_converges_to_exact(tiny_mlp, exact_spectrum):
    """
    Hutchinson-via-Lanczos trace (analysis/spectral_observables.py) vs. the
    exact trace from the materialized Hessian. `sum(weights * nodes)`
    reproduces v^T H v exactly for any m >= 1 -- it's a 1st-moment identity
    of Gauss quadrature -- so this axis is governed entirely by probe count
    (Monte Carlo variance of the Hutchinson estimator), not Lanczos depth.
    m is held fixed on purpose to isolate that.
    """
    model, params, loss = tiny_mlp
    exact_trace = exact_spectrum.sum().item()

    rel_errors = []
    for k in [10, 100, 500]:
        est = estimate_density(model, m=30, k=k, sigma=1.0, loss=loss, probe_seed=0)
        observed = compute_spectral_observables(est.probe_nodes, est.probe_weights, est.n_params)
        rel_errors.append(abs(observed.trace - exact_trace) / abs(exact_trace))

    assert rel_errors[0] > rel_errors[1] > rel_errors[2], rel_errors
    assert rel_errors[-1] < 0.02, f"trace rel error {rel_errors[-1]:.3%} at k=500, expected < 2%"


def test_top_eig_converges_with_lanczos_depth(tiny_mlp, exact_spectrum):
    """
    Single-probe Lanczos top Ritz value vs. exact lambda_max. Convergence
    here tracks Krylov depth (m), essentially independent of probe count --
    the opposite axis from the trace test above.
    """
    model, params, loss = tiny_mlp
    n = sum(p.numel() for p in params)
    exact_top = exact_spectrum.max().item()

    errors = []
    for m in [5, 10, 20, 40]:
        generator = torch.Generator().manual_seed(0)
        v0 = rademacher_probe(n, DEVICE, torch.float32, generator)
        eig = Eigenthings(model=model, m=m, sigma=1.0, params=params, device=DEVICE, loss=loss)
        nodes, _ = eig.lanczos_quadrature(v0)
        errors.append(abs(nodes.max().item() - exact_top))

    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 1e-4, f"top_eig error {errors[-1]:.2e} at m=40, expected < 1e-4"


def test_density_matches_exact_histogram(tiny_mlp, exact_spectrum):
    """
    SLQ-smoothed density vs. the exact eigenvalue histogram, broadened with
    the same Gaussian kernel width -- compares smoothed curves on a shared
    grid, not raw histograms vs. raw Ritz nodes (those are different
    discrete objects and aren't comparable directly).
    """
    model, params, loss = tiny_mlp
    sigma = 0.05 * (exact_spectrum.max().item() - exact_spectrum.min().item())

    est = estimate_density(model, m=100, k=300, sigma=sigma, loss=loss, probe_seed=0)

    # density_from_lanczos/gaussian_kernel take sigma as an explicit arg
    # (self.sigma from __init__ is unused by them), so m=1 here is a
    # throwaway -- this instance is only used for its density method.
    helper = Eigenthings(model=model, m=1, sigma=sigma, params=params, device=DEVICE, loss=loss)
    exact_weights = torch.full_like(exact_spectrum, 1.0 / exact_spectrum.numel())
    exact_density = helper.density_from_lanczos(exact_spectrum, exact_weights, est.t_grid, sigma)

    peak = exact_density.max().item()
    max_abs_diff = (est.density - exact_density).abs().max().item()
    assert max_abs_diff / peak < 0.03, (
        f"density max abs diff {max_abs_diff:.4f} = {max_abs_diff / peak:.2%} of peak {peak:.4f}"
    )


def test_negative_mass_recovered(tiny_mlp, exact_spectrum):
    """
    Eigenthings.hvp differentiates the real loss twice (no GGN/JᵀJ
    surrogate), so a random-init nonlinear net -- generically at an
    indefinite point, no bespoke saddle needed -- should show substantial
    negative spectral mass. If this were silently zero, that's the GGN/PSD
    contamination the checklist warns about.

    Unlike trace, this needs a deep Krylov space (m), not many probes: it
    depends on Lanczos resolving the *shape* of the spectral measure near
    the zero crossing, not just averaging out probe noise.
    """
    model, params, loss = tiny_mlp

    assert (exact_spectrum < 0).any(), "fixture has no negative curvature -- try a different seed"
    exact_neg_frac = (exact_spectrum < 0).float().mean().item()

    est = estimate_density(model, m=200, k=300, sigma=1.0, loss=loss, probe_seed=0)
    observed = compute_spectral_observables(est.probe_nodes, est.probe_weights, est.n_params)

    assert abs(observed.negative_mass - exact_neg_frac) < 0.03, (
        f"negative_mass {observed.negative_mass:.4f} vs exact {exact_neg_frac:.4f}"
    )


def test_reorthogonalization_prevents_ghosts(tiny_mlp, exact_spectrum):
    """
    With reorthogonalization off, loss of orthogonality among Krylov
    vectors lets Lanczos rediscover an already-converged extremal
    eigenvalue as a spurious duplicate ("ghost") Ritz value.

    A global near-duplicate count over the whole spectrum doesn't isolate
    this cleanly on this model -- the bulk is dense enough (1026
    eigenvalues, std ~0.2) that reorthogonalized Lanczos legitimately
    produces many close Ritz values there as m grows, which swamps the
    signal. The top eigenvalue is well-separated from its neighbor instead
    (gap ~0.08 vs. a 0.5%-of-top_eig window ~0.004), so ghosts show up
    unambiguously as a growing count of near-duplicate copies there.
    """
    model, params, loss = tiny_mlp
    n = sum(p.numel() for p in params)
    m = 100  # past top-eigenvalue convergence, giving loss of orthogonality room to matter

    generator = torch.Generator().manual_seed(0)
    v0 = rademacher_probe(n, DEVICE, torch.float32, generator)  # not mutated by lanczos(), safe to reuse

    eig_reorth = Eigenthings(model=model, m=m, sigma=1.0, params=params, device=DEVICE, loss=loss, orthogonalize=True)
    eig_noreorth = Eigenthings(model=model, m=m, sigma=1.0, params=params, device=DEVICE, loss=loss, orthogonalize=False)

    ritz_reorth = torch.linalg.eigvalsh(eig_reorth.lanczos(v0))
    ritz_noreorth = torch.linalg.eigvalsh(eig_noreorth.lanczos(v0))

    top = exact_spectrum.max().item()
    window = 0.005 * abs(top)
    near_top_reorth = (ritz_reorth > top - window).sum().item()
    near_top_noreorth = (ritz_noreorth > top - window).sum().item()

    assert near_top_reorth <= 2, f"reorthogonalized run should not show ghosts, got {near_top_reorth} copies"
    assert near_top_noreorth > near_top_reorth, (
        f"no-reorth ({near_top_noreorth}) should show more duplicates near top_eig than reorth "
        f"({near_top_reorth}) -- if this fails, orthogonalize=False isn't actually wired into lanczos()"
    )


def test_determinism_fixed_seed(tiny_mlp):
    """Same probe_seed -> bitwise-identical output, run to run."""
    model, params, loss = tiny_mlp
    est1 = estimate_density(model, m=30, k=50, sigma=1.0, loss=loss, probe_seed=42)
    est2 = estimate_density(model, m=30, k=50, sigma=1.0, loss=loss, probe_seed=42)

    assert torch.equal(est1.density, est2.density)
    assert torch.equal(est1.t_grid, est2.t_grid)
    for n1, n2 in zip(est1.probe_nodes, est2.probe_nodes):
        assert torch.equal(n1, n2)
