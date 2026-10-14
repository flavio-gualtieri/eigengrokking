# tests/test_hessian.py

"""
Hessian tooling (eigengrok/hessian/) against exact references, in float64.

The exact Hessian comes from tests/reference_hessian.py (one double-backward
pass per coordinate, independent of the torch.func code under test); the exact
Gauss-Newton matrix from J^T H_z J with J and H_z built by
torch.autograd.functional. Models are small enough (P ~ 500) to materialise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.autograd.functional import hessian as autograd_hessian
from torch.autograd.functional import jacobian as autograd_jacobian

from eigengrok.hessian.lanczos import lanczos
from eigengrok.hessian.operators import CurvatureOperators
from eigengrok.hessian.snapshot import fixed_eval_batch, hessian_snapshot
from eigengrok.hessian.spectrum import (
    STAT_NAMES,
    SpectralQuadrature,
    hutchinson_trace,
    rademacher_probes,
    slq,
    spectral_density,
    spectral_stats,
    spectral_stats_with_stderr,
    top_eigenpairs,
)
from eigengrok.models.mlp import MLP
from eigengrok.models.transformer import Transformer
from eigengrok.train.spectral_schedule import spectral_steps
from tests.reference_hessian import compute_full_hessian

CPU = torch.device("cpu")
F64 = torch.float64
N_SAMPLES, IN_DIM, N_CLASSES = 32, 10, 6

LOSSES = {
    "CrossEntropy": lambda z, y: F.cross_entropy(z, y),
    # Mean over N x C, as nn.MSELoss(); written out as in Task.compute_training_loss (F.mse_loss breaks under torch.func jvp-of-grad)
    "MSE": lambda z, y: ((z - F.one_hot(y, z.shape[-1]).to(z.dtype)) ** 2).mean(),
}


def _mlp_problem(loss_name: str, hidden=(30,), seed: int = 0):
    torch.manual_seed(seed)
    model = MLP(IN_DIM, hidden, N_CLASSES, device=CPU, activation_fn="Tanh").to(F64)
    x = torch.randn(N_SAMPLES, IN_DIM, dtype=F64)
    y = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    loss_fn = LOSSES[loss_name]
    ops = CurvatureOperators(model, x, y, loss_fn)
    H = compute_full_hessian(loss_fn(model(x), y), model).detach()
    return ops, H


def _exact_gauss_newton(ops: CurvatureOperators) -> torch.Tensor:
    z0 = ops.logits(ops.theta).detach()
    J = autograd_jacobian(lambda flat: ops.logits(flat).reshape(-1), ops.theta)          # [N*C, P]
    H_z = autograd_hessian(lambda zf: ops.loss_fn(zf.view_as(z0), ops.targets), z0.reshape(-1))  # [N*C, N*C]
    return J.T @ H_z @ J


def _materialise(op, n: int) -> torch.Tensor:
    return op(torch.eye(n, dtype=F64))  # rows are A e_i; A is symmetric


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


@pytest.fixture(scope="module")
def ce_problem():
    return _mlp_problem("CrossEntropy")


@pytest.fixture(scope="module")
def ce_spectrum(ce_problem):
    _, H = ce_problem
    return torch.linalg.eigvalsh(H)  # ascending


# --- operators ----------------------------------------------------------------

@pytest.mark.parametrize("loss_name", LOSSES)
def test_hvp_matches_exact_hessian(loss_name):
    ops, H = _mlp_problem(loss_name)
    V = torch.randn(5, ops.n_params, dtype=F64)
    assert _rel_err(ops.hvp(V), V @ H) < 1e-10


def test_hvp_matches_exact_hessian_on_transformer():
    """Forward-mode AD has to get through embeddings, LayerNorm, causal attention and GELU."""
    torch.manual_seed(0)
    p = 5
    model = Transformer(vocab_size=p + 1, output_dim=p, n_ctx=3, d_model=8, num_layers=1, num_heads=2,
                        device=CPU, d_mlp=8, activation_fn="GELU").to(F64)
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    x = torch.stack([a.flatten(), b.flatten(), torch.full((p * p,), p)], dim=1)
    y = (a.flatten() + b.flatten()) % p
    loss_fn = LOSSES["CrossEntropy"]
    ops = CurvatureOperators(model, x, y, loss_fn)
    H = compute_full_hessian(loss_fn(model(x), y), model).detach()
    V = torch.randn(3, ops.n_params, dtype=F64)
    assert _rel_err(ops.hvp(V), V @ H) < 1e-10


@pytest.mark.parametrize("loss_name", LOSSES)
def test_gauss_newton_plus_functional_is_the_hessian(loss_name):
    ops, H = _mlp_problem(loss_name)
    G_exact = _exact_gauss_newton(ops)
    G = _materialise(ops.gauss_newton, ops.n_params)
    H_f = _materialise(ops.functional, ops.n_params)

    assert _rel_err(G, G_exact) < 1e-10
    assert _rel_err(H_f, H - G_exact) < 1e-10
    assert _rel_err(G + H_f, H) < 1e-10
    # Both losses are convex in the logits, so G is PSD.
    assert torch.linalg.eigvalsh(G).min() > -1e-10 * G.abs().max()
    # The functional part is not negligible on this nonlinear net, so the checks above have teeth.
    assert H_f.abs().max() > 1e-3 * H.abs().max()


@pytest.mark.parametrize("loss_name", LOSSES)
def test_linear_model_has_no_functional_part(loss_name):
    """z is linear in theta, so d^2 z / d theta^2 = 0 and H = G exactly."""
    ops, H = _mlp_problem(loss_name, hidden=())
    H_f = _materialise(ops.functional, ops.n_params)
    G = _materialise(ops.gauss_newton, ops.n_params)
    assert H_f.abs().max() < 1e-12 * H.abs().max()
    assert _rel_err(G, H) < 1e-10


# --- Lanczos --------------------------------------------------------------------

def test_lanczos_top_eigenpairs_match_eigh(ce_problem):
    ops, H = ce_problem
    evals, evecs = torch.linalg.eigh(H)
    top_true, vecs_true = evals.flip(0)[:3], evecs.flip(1)[:, :3].T
    assert (top_true[:-1] - top_true[1:]).min() > 1e-3 * top_true[0], "fixture's top eigenvalues are too close"

    values, vectors, residuals = top_eigenpairs(ops.hvp, ops.n_params, n_top=3, m=80, seed=0, device=CPU, dtype=F64)
    np.testing.assert_allclose(values, top_true.numpy(), rtol=1e-8)
    overlaps = (vectors * vecs_true).sum(dim=1).abs()
    assert overlaps.min() > 1 - 1e-8, overlaps
    assert residuals.max() < 1e-6 * float(top_true[0])


def test_lanczos_basis_stays_orthonormal(ce_problem):
    """Full reorthogonalisation: without it, orthogonality is lost once top_eig converges and ghost copies appear."""
    ops, H = ce_problem
    start = rademacher_probes(2, ops.n_params, seed=0, device=CPU, dtype=F64)
    res = lanczos(ops.hvp, start, m=150, keep_basis=True)
    for i in range(2):
        Q = res.basis[i, : int(res.steps[i])]
        assert (Q @ Q.T - torch.eye(len(Q), dtype=F64)).abs().max() < 1e-10
    # No ghost copies of the top eigenvalue among the Ritz values.
    ritz = np.linalg.eigvalsh(res.tridiagonal(0))
    top = float(torch.linalg.eigvalsh(H).max())
    assert (ritz > top * (1 - 1e-6)).sum() == 1


def test_lanczos_stops_when_krylov_space_is_exhausted():
    ops, H = _mlp_problem("CrossEntropy", hidden=())  # P = 66, m > P
    start = rademacher_probes(1, ops.n_params, seed=0, device=CPU, dtype=F64)
    res = lanczos(ops.hvp, start, m=200)
    assert int(res.steps[0]) <= ops.n_params < 200
    (nodes,), (weights,) = res.quadrature()
    assert math.isclose(weights.sum(), 1.0, rel_tol=1e-12)
    v = start[0] / start[0].norm()
    for n in range(1, 8):  # the rule is now exact for every moment
        exact = float(v @ torch.linalg.matrix_power(H, n) @ v)
        assert math.isclose(float(np.sum(weights * nodes ** n)), exact, rel_tol=1e-8)


def test_batched_lanczos_matches_one_probe_at_a_time(ce_problem):
    ops, _ = ce_problem
    start = rademacher_probes(4, ops.n_params, seed=1, device=CPU, dtype=F64)
    batched = lanczos(ops.hvp, start, m=20)
    for i in range(4):
        single = lanczos(ops.hvp, start[i:i + 1], m=20)
        torch.testing.assert_close(batched.alpha[i], single.alpha[0], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(batched.beta[i], single.beta[0], rtol=1e-10, atol=1e-12)


# --- SLQ ------------------------------------------------------------------------

def test_quadrature_is_exact_for_low_moments(ce_problem):
    """An m-point Gauss rule integrates lambda^n exactly for n <= 2m - 1: sum_j w_j theta_j^n = v^T H^n v."""
    ops, H = ce_problem
    m = 10
    start = rademacher_probes(1, ops.n_params, seed=3, device=CPU, dtype=F64)
    (nodes,), (weights,) = lanczos(ops.hvp, start, m).quadrature()
    v = start[0] / start[0].norm()
    Hs = H / torch.linalg.matrix_norm(H, ord=2)  # keep powers O(1)
    scale = float(torch.linalg.matrix_norm(H, ord=2))
    for n in range(2 * m):
        exact = float(v @ torch.linalg.matrix_power(Hs, n) @ v)
        quad = float(np.sum(weights * (nodes / scale) ** n))
        assert abs(quad - exact) < 1e-9, (n, quad, exact)


def test_slq_moments_match_exact_traces(ce_problem, ce_spectrum):
    """
    The only error left in SLQ moments is the Monte Carlo error of the probe
    average (m=12 integrates lambda^n exactly up to n=23), so each estimate
    must sit within 4 standard errors of the exact tr(H^n)/P. k=400 keeps
    those standard errors under 5% of the exact value, so the check has teeth.
    """
    ops, _ = ce_problem
    q = slq(ops.hvp, ops.n_params, k=400, m=12, seed=0, device=CPU, dtype=F64)
    est, se = spectral_stats_with_stderr(q)
    for n in (1, 2, 3, 4):
        exact = float((ce_spectrum ** n).mean())
        assert se[f"moment_{n}"] < 0.05 * abs(exact), n
        assert abs(est[f"moment_{n}"] - exact) < 4 * se[f"moment_{n}"], (n, est[f"moment_{n}"], exact)
    assert abs(est["trace"] - float(ce_spectrum.sum())) < 4 * se["trace"]


def test_hutchinson_trace_shares_probes_with_slq(ce_problem, ce_spectrum):
    ops, _ = ce_problem
    trace, se = hutchinson_trace(ops.hvp, ops.n_params, k=50, seed=7, device=CPU, dtype=F64)
    q = slq(ops.hvp, ops.n_params, k=50, m=5, seed=7, device=CPU, dtype=F64)
    assert math.isclose(trace, spectral_stats(q)["trace"], rel_tol=1e-10)
    assert abs(trace - float(ce_spectrum.sum())) < 4 * se


def _exact_quadrature(spectrum: torch.Tensor) -> SpectralQuadrature:
    n = spectrum.numel()
    return SpectralQuadrature([spectrum.numpy()], [np.full(n, 1.0 / n)], n)


def test_stats_of_an_exact_spectrum():
    lam = torch.tensor([-0.6, -0.4, 0.0, 1.0, 2.0, 10.0], dtype=F64)
    tr = float(lam.sum())
    s = spectral_stats(_exact_quadrature(lam))
    assert set(s) == set(STAT_NAMES)
    assert math.isclose(s["trace"], tr)
    assert math.isclose(s["moment_2"], float((lam ** 2).mean()))
    assert s["top_eig"] == 10.0 and s["min_eig"] == -0.6
    assert math.isclose(s["top_over_trace"], 10 / tr)
    assert math.isclose(s["participation_ratio"], tr ** 2 / float((lam ** 2).sum()))
    assert math.isclose(s["stable_rank"], float((lam ** 2).sum()) / 100)
    assert math.isclose(s["normalized_moment_3"], float((lam ** 3).sum()) / tr ** 3)
    assert math.isclose(s["negative_mass"], 1 / 6)  # only -0.6 is below the margin -0.05 * 10
    assert math.isclose(s["n_outliers"], 2.0)       # 2 and 10 exceed 0.1 * 10
    # A flat |lambda| spectrum has effective rank equal to the number of non-zero eigenvalues.
    flat = spectral_stats(_exact_quadrature(torch.tensor([1.0, 1.0, -1.0, 1.0], dtype=F64)))
    assert math.isclose(flat["effective_rank"], 4.0, rel_tol=1e-9)


def test_slq_stats_match_exact_spectrum(ce_problem, ce_spectrum):
    """
    Extremes converge with Lanczos depth and must be essentially exact. The
    rest carry Monte Carlo error from the probe average: on this fixture the
    trace's standard error is ~7% of the trace at k=100 (indefinite spectrum,
    large off-diagonal mass), and ratios like tr(H)^2/tr(H^2) roughly double
    it. k=1000 brings that to ~2-5%, inside the 10% tolerance.
    """
    ops, _ = ce_problem
    exact = spectral_stats(_exact_quadrature(ce_spectrum))
    q = slq(ops.hvp, ops.n_params, k=1000, m=40, seed=0, device=CPU, dtype=F64)
    est = spectral_stats(q)
    assert math.isclose(est["top_eig"], exact["top_eig"], rel_tol=1e-8)
    assert math.isclose(est["min_eig"], exact["min_eig"], rel_tol=1e-8)
    for key in ("trace", "participation_ratio", "stable_rank", "top_over_trace",
                "spectral_entropy", "effective_rank", "n_outliers"):
        assert math.isclose(est[key], exact[key], rel_tol=0.1), (key, est[key], exact[key])
    assert abs(est["negative_mass"] - exact["negative_mass"]) < 0.03
    assert exact["negative_mass"] > 0.05, "fixture should have real negative curvature"


def test_density_grid_resolves_the_bandwidth():
    q = SpectralQuadrature([np.array([-1.0, 0.0, 3.0, 1e4])], [np.array([0.3, 0.4, 0.2, 0.1])], 4)
    grid, density = spectral_density(q, sigma_frac=0.01)
    sigma = 0.01 * (1e4 + 1)
    assert np.diff(grid).max() < sigma / 10
    assert math.isclose(np.trapezoid(density, grid), 1.0, rel_tol=1e-3)


# --- snapshots ------------------------------------------------------------------

def test_snapshot_is_deterministic(ce_problem):
    ops, _ = ce_problem
    a = hessian_snapshot(ops.model, ops.inputs, ops.targets, ops.loss_fn, m=10, k=4, probe_seed=5)
    b = hessian_snapshot(ops.model, ops.inputs, ops.targets, ops.loss_fn, m=10, k=4, probe_seed=5)
    assert a.stats == b.stats


def test_fixed_eval_batch():
    ds = torch.utils.data.TensorDataset(torch.arange(100).float()[:, None], torch.arange(100))
    x, y = fixed_eval_batch(ds, max_points=200, seed=0, device=CPU)
    assert torch.equal(y, torch.arange(100))  # small set: all of it, in order
    x1, y1 = fixed_eval_batch(ds, max_points=30, seed=0, device=CPU)
    x2, y2 = fixed_eval_batch(ds, max_points=30, seed=0, device=CPU)
    _, y3 = fixed_eval_batch(ds, max_points=30, seed=1, device=CPU)
    assert len(y1) == 30 and len(set(y1.tolist())) == 30
    assert torch.equal(y1, y2) and not torch.equal(y1, y3)


def test_spectral_steps():
    steps = spectral_steps(10_000, growth=1.15, every=1000)
    assert steps[:8] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert steps == sorted(set(steps)) and steps[-1] == 9000
    early = [s for s in steps if s < 1000]
    gaps = np.diff(early)
    assert (gaps[1:] >= gaps[:-1] - 1).all()  # geometric: gaps grow
    assert [s for s in steps if s >= 1000] == list(range(1000, 10_000, 1000))
