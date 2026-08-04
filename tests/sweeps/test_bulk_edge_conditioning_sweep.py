# tests/sweeps/test_bulk_edge_conditioning_sweep.py

"""
Committed driver for the extended-fixture (production-dynamic-range)
m-sweep documented in reports/spectral_validation.md. Previously this sweep
only existed as a scratch script and was never checked in -- the report
documented the exact construction (seed, eigenvalue list, n) precisely
enough to be "reproducible by description," but nothing in the repo could
actually regenerate the numbers. This module is the fix: it's real,
version-controlled code, marked `slow` (see pytest.ini) so it doesn't run as
part of the default fast suite (multi-minute runtime), but, from the repo
root, `python -m tests.sweeps.test_bulk_edge_conditioning_sweep` (needs `-m`
for the package-relative imports to resolve) reproduces every number in the
report's "bulk_edge / conditioning" section and the extended-fixture half of
"spectral_entropy / effective_rank", including the zero_frac diagnostic
column. Equivalently: `pytest tests/sweeps -m slow -s`.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.eigenthings import estimate_density
from analysis.spectral_observables import _pool_probes, compute_spectral_observables
from tests.sweeps.quadratic_fixture import exact_observables, make_fixture

M_GRID = (10, 20, 30, 50, 75, 100, 150, 200, 300)
K = 300
R = 3
SIGMA_FRAC = 0.01  # density/t_grid aren't read by any field swept here -- only affects the unused SpectralEstimate.density
ZERO_C = 1e-3      # fraction of |top_eig| below which a pooled Ritz node counts as "near zero"

FIELDS = ("spectral_entropy", "effective_rank", "bulk_edge", "conditioning")


def zero_frac(nodes: np.ndarray, weights: np.ndarray, top_eig: float, zero_c: float = ZERO_C) -> float:
    """
    Weighted fraction of pooled Ritz mass with |lambda| < zero_c * |top_eig|.

    Diagnostic for spectral_entropy's closed-form log(max(abs_nodes, eps))
    clip (analysis/spectral_observables.py): a high zero_frac means a
    meaningful share of the entropy cross-term is coming from that
    clipped/near-singular band rather than the bulk's actual shape -- a
    plausible, previously unchecked explanation for entropy's noisier,
    less-monotonic m-convergence compared to effective_rank in
    reports/spectral_validation.md (both read the same pooled distribution,
    but only entropy's cross-term involves a log of the node values).
    """
    threshold = zero_c * abs(top_eig)
    mask = np.abs(nodes) < threshold
    return float(weights[mask].sum() / weights.sum())


def run_sweep(model, loss, exact_spectrum, m_grid=M_GRID, k=K, r=R, sigma_frac=SIGMA_FRAC):
    exact = exact_observables(exact_spectrum)
    results = {name: {} for name in FIELDS}
    zero_fracs: dict[int, float] = {}

    for m in m_grid:
        per_field_errs = {name: [] for name in FIELDS}
        zf = []
        for rep in range(r):
            est = estimate_density(model, m=m, k=k, sigma_frac=sigma_frac, loss=loss, probe_seed=1000 * m + rep)
            obs = compute_spectral_observables(est.probe_nodes, est.probe_weights, est.n_params)
            nodes, weights = _pool_probes(est.probe_nodes, est.probe_weights)
            zf.append(zero_frac(nodes, weights, obs.top_eig))
            for name in FIELDS:
                exact_val, obs_val = getattr(exact, name), getattr(obs, name)
                per_field_errs[name].append(abs(obs_val - exact_val) / max(abs(exact_val), 1e-12))
        for name in FIELDS:
            arr = np.array(per_field_errs[name])
            results[name][m] = (float(arr.mean()), float(arr.std()))
        zero_fracs[m] = float(np.mean(zf))

    return exact, results, zero_fracs


def format_report(exact, results, zero_fracs, m_grid=M_GRID) -> str:
    lines = [
        f"exact: top_eig={exact.top_eig:.4f} bulk_edge={exact.bulk_edge:.4f} "
        f"conditioning={exact.conditioning:.4f} entropy={exact.spectral_entropy:.4f} "
        f"eff_rank={exact.effective_rank:.4f}",
        "".rjust(16) + "".join(f"m={m}".rjust(10) for m in m_grid),
    ]
    for name in FIELDS:
        lines.append(name.rjust(16) + "".join(f"{results[name][m][0]*100:9.2f}%" for m in m_grid))
    lines.append("zero_frac".rjust(16) + "".join(f"{zero_fracs[m]*100:9.4f}%" for m in m_grid))
    return "\n".join(lines)


@pytest.mark.slow
def test_bulk_edge_conditioning_converge_on_extended_fixture():
    """
    Regression guard for the bulk_edge/conditioning weighted-median/MAD fix
    (reports/spectral_validation.md, "bulk_edge / conditioning" section): on
    a fixture engineered to have a large bulk/outlier mass imbalance, the
    fixed estimator should land well under 5% by m=100. The pre-fix
    (unweighted np.median/MAD) estimator plateaued at ~30-50% even at
    m=300 -- an m-independent bias, not an under-convergence problem -- so
    this bound would catch a regression back to that behavior.
    """
    model, loss, exact_spectrum = make_fixture()
    exact = exact_observables(exact_spectrum)

    est = estimate_density(model, m=100, k=K, sigma_frac=SIGMA_FRAC, loss=loss, probe_seed=42)
    obs = compute_spectral_observables(est.probe_nodes, est.probe_weights, est.n_params)

    bulk_edge_err = abs(obs.bulk_edge - exact.bulk_edge) / abs(exact.bulk_edge)
    conditioning_err = abs(obs.conditioning - exact.conditioning) / abs(exact.conditioning)

    assert bulk_edge_err < 0.05, f"bulk_edge rel. error {bulk_edge_err:.2%} at m=100, expected <5%"
    assert conditioning_err < 0.05, f"conditioning rel. error {conditioning_err:.2%} at m=100, expected <5%"


@pytest.mark.slow
def test_full_sweep_regenerates_report_numbers(capsys):
    """
    Runs the full m-sweep and prints the table reports/spectral_validation.md
    is built from -- `pytest tests/sweeps -m slow -s` (or running this file
    directly) reproduces it. No assertions beyond test_bulk_edge_conditioning_
    converge_on_extended_fixture above; this exists so the numbers are one
    command away, not just described in prose.
    """
    model, loss, exact_spectrum = make_fixture()
    exact, results, zero_fracs = run_sweep(model, loss, exact_spectrum)
    with capsys.disabled():
        print("\n" + format_report(exact, results, zero_fracs))


if __name__ == "__main__":
    _model, _loss, _exact_spectrum = make_fixture()
    _exact, _results, _zero_fracs = run_sweep(_model, _loss, _exact_spectrum)
    print(format_report(_exact, _results, _zero_fracs))
