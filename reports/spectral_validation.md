# Spectral module validation: best `m`, `k` per observable

Validates `analysis/eigenthings.py` / `analysis/spectral_observables.py` against
an exact reference Hessian on a tiny MLP (input=10, hidden=(60,), output=6,
Tanh, ~1026 params, CrossEntropy, fixed seed), where the full Hessian can be
materialized column-by-column via HVP (`analysis/legacy/toy_eigenthings.py`).
Executable as `tests/test_eigenthings.py`; this doc adds the systematic
`m`/`k` sweeps that don't belong in a fast test suite (multi-minute runtime)
but are needed to answer "how much is enough."

Companion to `reports/spectral_methodology.txt` (the pipeline design). This
is the validation, not a re-description of the method.

## Headline result

The original production defaults (`configs/base.py`: `spectral_m=90`,
`spectral_k=16`) under-converged two of the six observables measured:

| | old production (m=90, k=16) | needed | verdict |
|---|---|---|---|
| **trace** | k=16 → ~17-19% rel. error (interpolated) | k≥100 for <5%, k≥~1000s for the ~1% original target | **under-converged** |
| **negative_mass (raw `< 0`)** | m=90 → measured ~17% relative bias (8pp absolute) at m=100 | m≥200 | **under-converged** |
| top_eig | m=90 ≫ 20 | m≥20 | fine |
| spectral_entropy / effective_rank | m=90 ≫ 20 | m≥20 | fine |
| density (plotted curve) | m=90 ≫ 20; k=16 a bit low | m≥20, k≥100 | mildly noisy, not biased |

`negative_mass`'s row is now historical: the raw `lambda < 0` threshold was
counting Lanczos noise right at the zero crossing as curvature evidence (see
below), and has been replaced with a margin-thresholded definition that
converges an order of magnitude faster in `m`. **Trace and (old)
negative_mass were governed by different axes than the other four
observables**, which is exactly why one `(m, k)` production default
couldn't serve all fields equally well — and why, even after the
negative_mass fix, `k` still needs to move independently of `m` for trace's
sake.

Based on these findings, `configs/base.py` now sets `spectral_m=100`,
`spectral_k=100` (was `m=90, k=16`) — see "Consequence for production
defaults" at the end of this doc.

**Caveat before using these numbers elsewhere:** all thresholds below are
measured on one ~1k-parameter toy MLP at one seed. The *qualitative*
per-observable sensitivity to `m` vs. `k` is a property of the Lanczos/SLQ
algorithm and should generalize; the *specific numeric* thresholds (m=20,
k=100, etc.) are shaped by this model's spectral gap and bulk density and
should not be copy-pasted onto the production transformer/mod-add runs
without their own check. Treat the production-default comparison above as a
directional flag worth following up, not a calibrated correction.

## Method

- **Exact reference for all 8 fields**, not just trace/top_eig/negative_mass:
  feed the full exact spectrum into `compute_spectral_observables` as a
  single pseudo-probe with uniform weight `1/n_params`. This reuses the real
  `spectral_entropy`/`effective_rank` histogram logic instead of re-deriving
  it by hand, so the approximate-vs-exact comparison is apples-to-apples.
  Sanity-checked against hand-derived values: trace 12.776217 vs. 12.776212,
  top_eig 0.872338 vs. 0.872338, negative_mass 0.482456 vs. 0.482456.
- **Independent probe draws per repeat.** An earlier pass at the trace
  convergence rate used one `probe_seed` per `k` and got a log-log slope
  nowhere near the theoretical −0.5 — traced to `estimate_density` reseeding
  a *fresh* generator per call and drawing sequentially, so `k=10` was
  literally a prefix of the `k=100`/`k=500` draws, not an independent
  realization. Every sweep below uses a distinct `probe_seed` per repeat.
  Re-running the trace sweep with `R=20` independent repeats per `k` gave a
  global log-log slope of **−0.511** (theory −0.5, within the ±0.05 band
  that would flag correlated probes) — confirms independent Rademacher
  sampling, no RNG bug.
- `effective_rank`/`spectral_entropy`/density sweeps use `R=3` independent
  repeats per grid point (mean ± std reported) — enough to keep single-draw
  noise from being mistaken for a convergence trend, without the cost of
  the `R=20` trace analysis.
- Determinism re-verified with `torch.use_deterministic_algorithms(True)`
  forced on (not just relying on CPU's incidental determinism): bitwise
  identical across repeated calls at fixed `probe_seed`.

## Per-observable results

### trace — governed by `k`, not `m`

`sum(weights * nodes)` reproduces `v^T H v` exactly for any `m ≥ 1` (a
1st-moment Gauss-quadrature identity, confirmed numerically: identical trace
at m=30 vs m=60). All convergence is Monte Carlo over probes.

RMS relative error, `R=20` independent repeats:

| k | 10 | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|---|
| RMS rel. error | 21.0% | 13.1% | 4.6% | 3.4% | 2.2% |

**Best: `m` irrelevant (use m≈30 for consistency with other calls in the same
pass), `k≥100`** for <5%; `k≥300` for <3.5%. Reaching the ~1% target from the
original spec would need `k` in the low thousands — the √k law is slow and
the "few hundred probes → ~1%" expectation was optimistic for this model.

### top_eig — governed by `m`, not `k`

Single generic probe, Krylov-depth sweep (algorithmic convergence, not an
average, so one realization is informative here unlike trace):

| m | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| abs. error | 2.9e-2 | 2.2e-4 | 3.0e-7 | 6.0e-8 (float32 floor) |

**Best: `m≥20`, `k=1`.**

### negative_mass — redefined mid-validation; margin-thresholding fixes the convergence problem

**Original definition** (`sum(weights[nodes < 0])`, raw zero threshold):
governed by `m`, needed the deepest Krylov space of any field measured.

| m | 30 | 60 | 100 | 200 |
|---|---|---|---|---|
| estimate (exact = 0.4825) | 0.598 | 0.554 | 0.403–0.407 | 0.482–0.486 |

`k` barely mattered at fixed `m`. Convergence was **non-monotonic** — m=100
was worse than m=60 — so intermediate depths weren't a safe interpolation;
only m=200 reliably landed within ~1pp of exact among tested values.

Root cause, found by inspecting the exact spectrum directly: 193 of 1026
exact eigenvalues on this fixture sit in `[-0.0436, 0)` — within 5% of
`|top_eig|` of zero. A raw `< 0` threshold counts every one of those as
"negative curvature," but Lanczos needs a lot of Krylov depth to correctly
resolve the sign of Ritz values that close to the crossing; small-m runs
were mostly measuring Lanczos noise in that band, not real signal.

**Fix:** redefined as margin-thresholded (`analysis/spectral_observables.py`,
`SpectralObservables.negative_mass`): `N_epsilon = P(lambda < -epsilon)`,
`epsilon = margin_c * |top_eig|` (default `margin_c=0.05`, shared default
with `resolution_frac` though independently tunable). This is an exact
reference-value change, not just an estimator change — exact
`negative_mass` on this fixture drops from 0.4825 (raw) to **0.2943**
(margin-thresholded), since it now excludes that whole near-zero band by
construction rather than trying to resolve it.

m-sweep (k=300, R=3, mean abs. error vs. exact 0.2943):

| m | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|
| abs. error | 0.114 | 0.032 | 0.007 | **0.001** | 0.004 | 0.002 | 0.002 | 0.001 |

k-sweep (m=100, R=3): flat at 0.001–0.004 across k=10 to 500 — `k` barely
matters here either, same pattern as entropy/rank/top_eig.

**Best: `m≥30` (m=50 for full noise-floor accuracy), `k≥30`.** An order of
magnitude cheaper than the raw definition's `m≥200`, and it's a more honest
number besides: it no longer reports near-zero Lanczos noise as evidence of
negative curvature. The GGN/PSD-contamination check this field exists for
(does negative curvature show up at all on a net at a saddle) doesn't need
the noise band anyway — a margin-thresholded 29% negative mass is exactly as
strong a "this is not PSD" signal as 48% was.

### spectral_entropy / effective_rank — governed by `m`, converges fast

Both read the *shape* of the pooled `|lambda|` histogram, same family as
negative_mass — but converge an order of magnitude faster.

m-sweep (k=300, R=3, mean rel. error):

| m | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|
| entropy | 10.8% | 0.24% | 0.41% | 0.68% | 0.26% | 0.30% | 0.31% | 0.23% |
| eff. rank | 19.5% | 0.49% | 0.83% | 1.36% | 0.52% | 0.60% | 0.63% | 0.46% |

k-sweep (m=150, R=3): flat at 0.19–0.56% (entropy) / 0.37–1.12% (rank) across
`k` = 10 to 1000 — once `m` clears the threshold, more probes buy almost
nothing.

**Best: `m≥20` (use m=30 for margin), `k≥30`.** Why so much faster than
negative_mass despite being the same "resolve the distribution shape" family:
the entropy histogram bins at a coarse 5%-of-top_eig width, so it only needs
Ritz mass in roughly the right bin; negative_mass needs the exact sign
relative to zero, an arbitrarily fine distinction right at the crossing that
coarse binning can't average over.

### density (SLQ vs. exact histogram, matched Gaussian broadening) — governed by `m` first, mildly helped by `k`

m-sweep (k=300, R=3, mean max-abs-diff / peak):

| m | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|
| density error | 5.0% | 0.27% | 0.26% | 0.36% | 0.26% | 0.30% | 0.25% | 0.32% |

k-sweep (m=150, R=3): 1.05% (k=10) → 1.06% (k=30) → 0.48% (k=100) → 0.41%
(k=200) → 0.33% (k=500) → 0.24% (k=1000) — unlike entropy/rank, density keeps
improving mildly with more probes past the point `m` saturates, likely
because it needs smooth coverage of the whole `t_grid`, not just the right
histogram bin.

**Best: `m≥20` (use m=30), `k≥100`** for a comfortably-sub-1% match.

## Summary table

| observable | governing axis | best `m` | best `k` |
|---|---|---|---|
| trace | k | 30 (m irrelevant) | ≥100 (≥300 for tighter) |
| top_eig | m | ≥20 | 1 |
| negative_mass (margin-thresholded) | m | ≥30 (50 for full margin) | ≥30 |
| spectral_entropy | m | ≥20 (30 w/ margin) | ≥30 |
| effective_rank | m | ≥20 (30 w/ margin) | ≥30 |
| density | m, then k | ≥20 (30 w/ margin) | ≥100 |

A single `(m, k)` that comfortably serves everything measured here: **m=50,
k=300** — `m` is now set by top_eig/negative_mass/entropy/rank/density
(all clear by ~m=30-50, nothing near the old m=200 floor), `k` is still set
by trace alone.

## Consequence for production defaults

Before the negative_mass fix, production (`m=90, k=16`) was under-converged
on two fields for two different reasons (`m` too shallow for negative_mass,
`k` too small for trace). After the fix, `m=90` already clears every
remaining `m`-bound field with room to spare — the negative_mass fix removed
the only reason `m` needed to be large. `k=16` remains the real gap; trace's
√k convergence means it needed to move a lot, not a little.

`configs/base.py` now sets **`spectral_m=100`, `spectral_k=100`**: `m` bumped
modestly (90→100, mostly a round-number safety margin — 90 was already
sufficient post-fix), `k` bumped substantially (16→100) to bring trace from
~17-19% down to ~5% relative error. This is a deliberate compromise, not the
"best k" row above (k≥300 for tighter trace accuracy, and the original ~1%
target needs k in the thousands): `k` multiplies the per-checkpoint HVP cost
directly (cost ≈ m·k HVPs), and this runs repeatedly over the course of
training via `SpectralSchedule`, not once — k=100 is a ~6x cost increase
over the old k=16 default; k=300 would be ~19x. If trace accuracy tighter
than ~5% turns out to matter for a specific analysis, that's a reason to
bump `spectral_k` further for that run specifically, with the compute-cost
tradeoff in mind, rather than raising the global default further.

## Source

Sweep scripts run interactively, not checked in (scratch, reproducible from
this doc's numbers and `tests/test_eigenthings.py`'s fixtures). The two
sub-checks that *are* checked in as regression tests:
`test_effective_rank_and_entropy_converge_with_lanczos_depth` and the
existing `test_top_eig_converges_with_lanczos_depth` /
`test_negative_mass_recovered` / `test_trace_converges_to_exact` /
`test_density_matches_exact_histogram` in `tests/test_eigenthings.py`.
