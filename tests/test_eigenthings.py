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


@pytest.fixture(scope="module")
def exact_observables(exact_spectrum):
    """
    Exact reference for *all* SpectralObservables fields -- not just the
    trace/top_eig/negative_mass computed by hand elsewhere in this file --
    by feeding the full exact spectrum in as a single pseudo-probe with
    uniform weight 1/n_params through the same compute_spectral_observables()
    used on the approximate side. This reuses the real spectral_entropy/
    effective_rank histogram logic rather than re-deriving it separately, so
    the comparison stays apples-to-apples.
    """
    n = exact_spectrum.numel()
    return compute_spectral_observables(
        probe_nodes=[exact_spectrum],
        probe_weights=[torch.full_like(exact_spectrum, 1.0 / n)],
        n_params=n,
    )


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
        est = estimate_density(model, m=30, k=k, sigma_frac=1.0, loss=loss, probe_seed=0)
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

    est = estimate_density(model, m=100, k=300, sigma_frac=0.05, loss=loss, probe_seed=0)
    sigma = est.sigma  # realized absolute bandwidth (sigma_frac * cheap top_eig pre-pass estimate)

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


def test_negative_mass_recovered(tiny_mlp, exact_spectrum, exact_observables):
    """
    Eigenthings.hvp differentiates the real loss twice (no GGN/JᵀJ
    surrogate), so a random-init nonlinear net -- generically at an
    indefinite point, no bespoke saddle needed -- should show substantial
    negative spectral mass. If this were silently zero, that's the GGN/PSD
    contamination the checklist warns about.

    negative_mass is margin-thresholded (mass below -margin_c * |top_eig|,
    not raw lambda < 0) specifically so this isn't chasing Lanczos noise
    right at the zero crossing -- see analysis/spectral_observables.py. This
    fixes what used to be the worst convergence in the suite: the raw
    lambda < 0 definition needed m>=200 to land within 1pp of exact (193 of
    1026 exact eigenvalues on this fixture sit in the now-excluded
    [-epsilon, 0) noise band); margin-thresholded, m=30 already gets within
    ~1% and m=50 is noise-floor accurate. See reports/spectral_validation.md.
    """
    model, params, loss = tiny_mlp

    assert (exact_spectrum < 0).any(), "fixture has no negative curvature -- try a different seed"

    est = estimate_density(model, m=50, k=100, sigma_frac=1.0, loss=loss, probe_seed=0)
    observed = compute_spectral_observables(est.probe_nodes, est.probe_weights, est.n_params)

    assert abs(observed.negative_mass - exact_observables.negative_mass) < 0.02, (
        f"negative_mass {observed.negative_mass:.4f} vs exact {exact_observables.negative_mass:.4f}"
    )


def test_effective_rank_and_entropy_converge_with_lanczos_depth(tiny_mlp, exact_observables):
    """
    spectral_entropy/effective_rank also read the *shape* of the pooled
    spectral measure, the same family as negative_mass -- but converge with
    m an order of magnitude faster (~20 vs ~200), confirmed by a
    3-repeat/independent-seed sweep before picking these thresholds (see
    reports/spectral_validation.md).

    spectral_entropy is the closed-form log(n_params * sum_j w_j|theta_j|)
    - sum_j w_j|theta_j| log|theta_j| / sum_j w_j|theta_j| computed directly
    off the raw (node, weight) atoms -- no binning, so no cap on
    effective_rank = exp(spectral_entropy) (it lands around n_params/2 on
    this fixture, not the old histogram's hard ceiling of 20). Because
    effective_rank is entropy's exponential, the same entropy error that
    clears 2% turns into a few points more once exponentiated -- hence the
    looser effective_rank tolerance below.
    """
    model, params, loss = tiny_mlp

    entropy_errors = []
    rank_errors = []
    for m in [10, 20, 30, 50]:
        est = estimate_density(model, m=m, k=300, sigma_frac=1.0, loss=loss, probe_seed=0)
        observed = compute_spectral_observables(est.probe_nodes, est.probe_weights, est.n_params)
        entropy_errors.append(
            abs(observed.spectral_entropy - exact_observables.spectral_entropy) / abs(exact_observables.spectral_entropy)
        )
        rank_errors.append(
            abs(observed.effective_rank - exact_observables.effective_rank) / abs(exact_observables.effective_rank)
        )

    assert entropy_errors[0] > entropy_errors[-1], entropy_errors
    assert rank_errors[0] > rank_errors[-1], rank_errors
    assert entropy_errors[-1] < 0.02, f"spectral_entropy rel error {entropy_errors[-1]:.3%} at m=50, expected < 2%"
    assert rank_errors[-1] < 0.05, f"effective_rank rel error {rank_errors[-1]:.3%} at m=50, expected < 5%"


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
    """
    Same probe_seed -> bitwise-identical output, run to run, with
    torch.use_deterministic_algorithms(True) forced on. Without forcing it,
    a bitwise match on CPU could pass "by luck" (CPU ops are mostly, but not
    universally, deterministic by default) while a nondeterministic kernel
    is still reachable -- which would show up downstream as
    checkpoint-to-checkpoint jitter that looks like a weak spectral signal.
    Forcing the flag makes PyTorch raise instead of silently taking a
    nondeterministic path.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model, params, loss = tiny_mlp
        est1 = estimate_density(model, m=30, k=50, sigma_frac=1.0, loss=loss, probe_seed=42)
        est2 = estimate_density(model, m=30, k=50, sigma_frac=1.0, loss=loss, probe_seed=42)

        assert torch.equal(est1.density, est2.density)
        assert torch.equal(est1.t_grid, est2.t_grid)
        for n1, n2 in zip(est1.probe_nodes, est2.probe_nodes):
            assert torch.equal(n1, n2)
    finally:
        torch.use_deterministic_algorithms(prev)
