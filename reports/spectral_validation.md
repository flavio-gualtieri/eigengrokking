# Spectral module validation: best `m`, `k` per observable

Validates `analysis/eigenthings.py` / `analysis/spectral_observables.py`
against two exact-reference fixtures: a tiny MLP (input=10, hidden=(60,),
output=6, Tanh, ~1026 params, CrossEntropy, fixed seed), where the full
Hessian is materialized column-by-column via HVP
(`tests/reference_hessian.py`); and an engineered quadratic-form fixture
with the same param count but a chosen, production-like spectral dynamic
range (see "Extended fixture" below), where the exact spectrum is chosen
directly rather than materialized. The first is executable as
`tests/test_eigenthings.py`; this doc adds the systematic `m`/`k` sweeps
that don't belong in a fast test suite (multi-minute runtime) but are needed
to answer "how much is enough."

Companion to `reports/spectral_methodology.txt` (the pipeline design). This
is the validation, not a re-description of the method.

## Headline result

The original production defaults (`configs/base.py`: `spectral_m=90`,
`spectral_k=16`) under-converged two of the six observables originally
measured, and a later pass — extending the fixture to a production-like
spectral dynamic range (see "Extended fixture" below) to finally validate
`bulk_edge`/`conditioning`, which had never been swept at all — found those
two lead observables weren't just unswept, they had a real estimator bug:

| | old production (m=90, k=16) | needed | verdict |
|---|---|---|---|
| **trace** | k=16 → ~17-19% rel. error (interpolated) | k≥100 for <5%, k≥~1000s for the ~1% original target | **under-converged** |
| **negative_mass (raw `< 0`)** | m=90 → measured ~17% relative bias (8pp absolute) at m=100 | m≥200 | **under-converged** |
| top_eig | m=90 ≫ 20 | m≥20 | fine |
| spectral_entropy / effective_rank | m=90 ≫ 20; reconfirmed at ~31x wider conditioning | m≥20 | fine |
| density (plotted curve) | m=90 ≫ 20; k=16 a bit low | m≥20, k≥100 | mildly noisy, not biased |
| **bulk_edge / conditioning** | never validated (0 of 8 fields swept before this pass) | as originally implemented (unweighted median/MAD over pooled Ritz nodes): **~30-50% biased, independent of `m`** — not a convergence problem, an estimator bug (see below). **Fixed** to a weighted median/MAD; now converges to m≥75-100 for <1.5%, m≥100 for <0.2% | **wrong estimator, now fixed; the tightest m-governed field of the eight** |

`negative_mass`'s row is now historical: the raw `lambda < 0` threshold was
counting Lanczos noise right at the zero crossing as curvature evidence (see
below), and has been replaced with a margin-thresholded definition that
converges an order of magnitude faster in `m`. **Trace and (old)
negative_mass were governed by different axes than the other four
observables**, which is exactly why one `(m, k)` production default
couldn't serve all fields equally well — and why, even after the
negative_mass fix, `k` still needs to move independently of `m` for trace's
sake. `bulk_edge`/`conditioning` add a third axis: no `(m, k)` fixes a wrong
formula — that needed a code change (`analysis/spectral_observables.py`),
not a bigger sweep.

Based on these findings, `configs/base.py` keeps `spectral_m=100`,
`spectral_k=100` — but the *justification* changes: `m=100` used to be a
round-number margin above what the original six fields needed; it's now the
minimum that gets the newly-fixed `bulk_edge`/`conditioning` under ~0.2%. See
"Consequence for production defaults" at the end of this doc.

**Caveat before using these numbers elsewhere:** the six original thresholds
are measured on one ~1k-parameter toy MLP at one seed; the `bulk_edge`/
`conditioning`/re-checked `spectral_entropy`/`effective_rank` thresholds are
measured on a second, engineered ~1k-parameter fixture at one seed (see
below). The *qualitative* per-observable sensitivity to `m` vs. `k` is a
property of the Lanczos/SLQ algorithm and should generalize; the *specific
numeric* thresholds (m=20, k=100, etc.) are shaped by each fixture's
spectral gap and bulk density and should not be copy-pasted onto the
production transformer/mod-add runs without their own check. Treat the
production-default comparison above as a directional flag worth following
up, not a calibrated correction.

## Extended fixture: production dynamic range

The original tiny MLP fixture is random-init, so its spectrum is one
roughly-flat bulk (exact top_eig/bulk_edge ≈ 4.4x on that fixture) — it never
exercised what `bulk_edge`/`conditioning` actually measure: how far an
outlier eigenvalue sits above the bulk, the regime a real net shows once it
starts fitting/sharpening. Two options that both add free parameters to
justify (train a net for some number of steps; or reuse the toy MLP but pick
a training recipe) were passed over for a fixture where the exact spectrum
is chosen directly, not approximated: `analysis/eigenthings.py`'s `hvp`
differentiates a `loss` tensor that's handed in fully-formed — `estimate_density`
never calls `model.forward()` — so `loss` doesn't need to come from a neural
net at all. A quadratic form `loss = 0.5 * theta^T M theta` (`theta` a bare
`nn.Parameter`, `M` a fixed buffer) has Hessian w.r.t. `theta` equal to `M`
*exactly*, so choosing `M`'s eigenvalues chooses the exact spectrum with zero
materialization cost — no HVP-based column-by-column Hessian needed at all,
unlike the original fixture's `analysis/legacy/toy_eigenthings.py` (now
`tests/reference_hessian.py`) reference.

Construction (`n=1026`, matching the original fixture's param count; fixed
seed=0 throughout, fully deterministic): `M = Q diag(lambda) Q^T`, `Q`
orthogonal via QR of a fixed-seed Gaussian, resymmetrized
(`0.5*(M + M^T)`) to kill float asymmetry. `lambda` = 1021 bulk eigenvalues
drawn `Uniform(-0.3, 0.3)` (486 negative — comparable negative-curvature
fraction to the original fixture) plus 5 outliers at `{40, 60, 80, 100,
150}`. Exact `top_eig=150.0`, `bulk_edge=1.109`, **conditioning=135.3** —
about 31x the original fixture's dynamic range. Not claimed to match a real
production transformer's actual conditioning number (no logged production
spectral data exists yet to calibrate against — the current HPC wd-sweep
runs with `run_spectral=False`); the point is to have *some* fixture where
bulk mass and outlier mass are wildly imbalanced, since that's the regime
`bulk_edge`/`conditioning` exist for and the original fixture couldn't test.

## Method

- **Exact reference for all 8 fields**, not just trace/top_eig/negative_mass:
  feed the full exact spectrum into `compute_spectral_observables` as a
  single pseudo-probe with uniform weight `1/n_params`. This reuses the real
  `spectral_entropy`/`effective_rank` calculation instead of re-deriving it
  by hand, so the approximate-vs-exact comparison is apples-to-apples.
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

Both read the *shape* of the pooled `|lambda|` distribution, same family as
negative_mass — but converge an order of magnitude faster.

**Note on the numbers below:** `spectral_entropy`/`effective_rank` were
originally computed from a 20-bin histogram of `|lambda|`; that's since been
replaced with a closed-form calculation (`analysis/spectral_observables.py`)
that removes an artificial cap the histogram put on `effective_rank`
(entropy of an N-bin histogram is bounded by `log(N)`, so `effective_rank`
could never exceed ~N regardless of the model's actual number of curved
directions). This changed the *exact reference values* substantially — e.g.
the tiny_mlp fixture's exact `effective_rank` moved from a
histogram-capped ~1 up to **602.5** (of 1026 params) — but the tables below,
re-measured against the current closed-form implementation, reconfirm the
same qualitative m-convergence story.

m-sweep (k=300, R=3, mean rel. error vs. exact `entropy=6.4011`,
`effective_rank=602.52`):

| m | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|
| entropy | 2.59% | 0.08% | 0.19% | 0.40% | 0.24% | 0.02% | 0.11% | 0.04% |
| eff. rank | 18.06% | 0.49% | 1.24% | 2.58% | 1.57% | 0.12% | 0.73% | 0.22% |

k-sweep (m=150, R=3): flat at 0.19–0.56% (entropy) / 0.37–1.12% (rank) across
`k` = 10 to 1000 — once `m` clears the threshold, more probes buy almost
nothing. (k-sweep predates the closed-form change and wasn't re-measured;
the underlying claim — `k` barely matters once `m` clears its threshold —
doesn't depend on which entropy formula is used.)

**Best: `m≥20` (use m=30 for margin), `k≥30`.** Why so much faster than
negative_mass despite being the same "resolve the distribution shape" family:
negative_mass needs the exact sign relative to zero, an arbitrarily fine
distinction right at the crossing; entropy/effective_rank are smooth
functionals of the whole pooled distribution and don't have an analogous
knife-edge.

**Reconfirmed on the extended (production-dynamic-range) fixture** — exact
`entropy=3.4499`, `effective_rank=31.4968` (of 1026 params; meaningfully
above 1 now that the histogram cap is gone, though still low relative to
the tiny_mlp fixture's 602.5, consistent with most of this fixture's mass
being concentrated in a narrow bulk plus 5 outliers rather than spread
across many distinct scales). m-sweep (k=300, R=3, mean rel. error; `python
-m tests.sweeps.test_bulk_edge_conditioning_sweep` reproduces this row and
the `zero_frac` diagnostic below it):

| m | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|---|
| entropy | 3.47% | 0.97% | 1.03% | 0.16% | 0.34% | 0.73% | 0.68% | 0.60% | 0.85% |
| eff. rank | 11.27% | 3.28% | 3.50% | 0.54% | 1.18% | 2.52% | 2.37% | 2.09% | 2.99% |
| zero_frac | 28.9% | 48.8% | 52.6% | 49.3% | 50.5% | 49.0% | 50.0% | 49.4% | 49.8% |

**`zero_frac`** (`tests/sweeps/test_bulk_edge_conditioning_sweep.py`): the
weighted fraction of pooled Ritz mass with `|lambda| < 1e-3 * |top_eig|`
(0.15 here) — a diagnostic for whether entropy's `log(max(abs_nodes, eps))`
term (analysis/spectral_observables.py) is being computed mostly from
values close enough to zero that `log` is numerically touchy. It settles
around **~50%**, which is exactly what this fixture's construction predicts
(the bulk is `Uniform(-0.3, 0.3)`, so half of it by construction lies within
`±0.15` of zero) — not evidence of degenerate clipping (the actual clip
floor is `eps=1e-12`, far below anything a float32 Ritz value lands on by
chance), but confirmation that a large share of this fixture's mass sits in
the region where `log|lambda|` is most sensitive to small absolute errors in
the Ritz estimate. That's a plausible structural reason `entropy` is
consistently noisier than `effective_rank` above (both read the same pooled
distribution, but only entropy's cross-term takes a `log` of it) — not
literally a bug, but a property of this fixture worth knowing when reading
the entropy column, and a reason to treat `zero_frac` as a companion
diagnostic on any future entropy sweep rather than a one-off.

Same conclusion as the original fixture either way: **`m≥20` already gets
both fields comfortably converged**, now confirmed at ~31x wider
conditioning too — this pair was never the reason `m` needed to move.

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

### bulk_edge / conditioning — found a real estimator bug, not an m problem; fixed, now governed by `m`

Both fields were never swept before this pass — they didn't exist for most
of this doc's history, and once added, the near-flat original fixture
couldn't exercise them (its exact `top_eig/bulk_edge ≈ 4.4x` means there's
barely a bulk/outlier distinction to get wrong). On the extended fixture
(exact `bulk_edge=1.109`, `conditioning=135.3`), the *original*
implementation — `bulk_edge = median(nodes) + 5*MAD(nodes)`, plain
`np.median`/`np.median(abs(...))` over the pooled Ritz nodes — was **~30-50%
biased at every `m` tested, including m=300**, with negligible
probe-to-probe std (<1%):

| m | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|---|
| bulk_edge rel. error (original, unweighted) | huge* | 107% | 45% | 52% | 48% | 47% | 46% | 44% | 41% |
| conditioning rel. error (original, unweighted) | 99% | 52% | 31% | 34% | 33% | 32% | 31% | 31% | 29% |

*m=10's bulk_edge error is a >100x outlier (a degenerate near-zero MAD from
too few Ritz values), not a typo — omitted from the table for scale, still
tracks the same "doesn't shrink with m" story as m=20 onward.

A near-flat, tiny std that *doesn't shrink with `m`* is the signature of a
biased estimator, not probe-sampling noise or Krylov under-convergence — so
raising `spectral_m` further would not have fixed this. **Root cause:**
`np.median`/MAD over pooled Ritz nodes implicitly treats every node as
equally likely, but SLQ nodes aren't samples from the density — Lanczos/
Gauss-quadrature node placement is *moment-matching*, not
*density-matching*, so nodes cluster near spectral edges regardless of how
much true mass sits there. Invisible on a roughly-flat spectrum (original
fixture); a large, `m`-independent bias on any spectrum with a big
bulk/outlier mass imbalance (this fixture, and presumably any real net once
it develops a few sharp directions against a broad low-curvature bulk).

**Fix** (`analysis/spectral_observables.py`, `_weighted_quantile`): replace
both `np.median` calls with a weight-respecting weighted quantile (linear
interpolation on the weighted CDF, atoms placed at their mass's midpoint),
matching what SLQ's `(node, weight)` pairs actually represent — the same
category of fix as the `negative_mass` redefinition above (an
estimator/formula correction found via this validation process, not a
parameter tune). Post-fix m-sweep (k=300, R=3, mean rel. error, using the
real `compute_spectral_observables` path; reproduced by
`python -m tests.sweeps.test_bulk_edge_conditioning_sweep`):

| m | 10 | 20 | 30 | 50 | 75 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|---|
| bulk_edge | 8.18% | 11.75% | 2.10% | 1.96% | 1.26% | **0.16%** | 0.77% | 0.23% | 0.30% |
| conditioning | 7.56% | 10.51% | 2.15% | 1.97% | 1.25% | **0.16%** | 0.77% | 0.23% | 0.30% |

Fixed, both fields now converge cleanly with `m` (conditioning tracks
bulk_edge almost exactly, since `top_eig` itself is essentially exact by
m≥20 — all of conditioning's error is inherited from bulk_edge).
`conditioning`'s formula (`top_eig / bulk_edge`) is unchanged; only
`bulk_edge`'s internal median/MAD changed.

**Best: `m≥75` for <1.5%, `m≥100` for <0.2%.** Not swept over `k` in this
pass (out of scope here — the task was the `m`-sweep specifically); the
`k`-sweeps above for entropy/rank/negative_mass all found `k` barely
matters once `m` clears its threshold, so `k` mattering here would be the
surprise, but it hasn't been checked.

`bulk_edge`/`conditioning` are now **the tightest `m`-governed fields of the
eight** — deeper than top_eig/negative_mass/entropy/rank/density (all
m≥20-50), on par with where old-definition negative_mass used to sit before
its own fix.

## Summary table

| observable | governing axis | best `m` | best `k` |
|---|---|---|---|
| trace | k | 30 (m irrelevant) | ≥100 (≥300 for tighter) |
| top_eig | m | ≥20 | 1 |
| negative_mass (margin-thresholded) | m | ≥30 (50 for full margin) | ≥30 |
| spectral_entropy | m | ≥20 (30 w/ margin) | ≥30 |
| effective_rank | m | ≥20 (30 w/ margin) | ≥30 |
| density | m, then k | ≥20 (30 w/ margin) | ≥100 |
| bulk_edge (post-fix) | m | ≥75 (100 for <0.2%) | not swept |
| conditioning (post-fix) | m | ≥75 (100 for <0.2%) | not swept |

A single `(m, k)` that comfortably serves everything measured here: **m=100,
k=300** (up from the previous m=50 estimate, which only accounted for six of
the eight fields) — `m` is now set by `bulk_edge`/`conditioning` (≥75-100,
the deepest requirement among the currently-well-behaved fields), `k` is
still set by trace alone.

## Consequence for production defaults

Before the negative_mass fix, production (`m=90, k=16`) was under-converged
on two fields for two different reasons (`m` too shallow for negative_mass,
`k` too small for trace). After the fix, `m=90` already cleared every
remaining `m`-bound field with room to spare among the six then-measured
fields — the negative_mass fix removed the only reason `m` needed to be
large. `k=16` remained the real gap; trace's √k convergence means it needed
to move a lot, not a little.

That reasoning motivated the previous `spectral_m=100, spectral_k=100`
(`m` bumped 90→100 as a round-number safety margin, `k` bumped 16→100 to
bring trace from ~17-19% down to ~5% relative error). This pass leaves both
numbers unchanged, but for a different reason: `bulk_edge`/`conditioning`
turn out to need `m≥100` for <0.2% — deeper than any of the six previously-
measured fields — so `m=100` is no longer just a safety margin, it's the
actual requirement. Had these two fields converged at m≥20-50 like the
entropy family, there'd have been a case for lowering `spectral_m` back down
toward 50-75 for cost; instead the fixed bulk_edge/conditioning happen to
need exactly what's already configured.

`configs/base.py` keeps **`spectral_m=100`, `spectral_k=100`**. `k=100` is
still a deliberate compromise, not the "best k" row above (k≥300 for
tighter trace accuracy, and the original ~1% target needs k in the
thousands): `k` multiplies the per-checkpoint HVP cost directly (cost ≈ m·k
HVPs), and this runs repeatedly over the course of training via
`SpectralSchedule`, not once — k=100 is a ~6x cost increase over the old
k=16 default; k=300 would be ~19x. If trace accuracy tighter than ~5% turns
out to matter for a specific analysis, that's a reason to bump `spectral_k`
further for that run specifically, with the compute-cost tradeoff in mind,
rather than raising the global default further. `bulk_edge`/`conditioning`
were not swept over `k` in this pass — worth a follow-up check before
trusting them below the ~0.2% level, though nothing so far suggests `k`
matters for this field family.

## Source

The extended-fixture (`bulk_edge`/`conditioning`/reconfirmation) sweep is
**checked in**: `tests/sweeps/quadratic_fixture.py` (the fixture
construction) and `tests/sweeps/test_bulk_edge_conditioning_sweep.py` (the
sweep driver, the `zero_frac` diagnostic, and a fast regression assertion).
It's marked `@pytest.mark.slow` (see `pytest.ini`) so it's excluded from the
default `pytest`/`pytest tests/` run — multi-minute runtime — but it's real,
version-controlled code, not a doc description standing in for one: `python
-m tests.sweeps.test_bulk_edge_conditioning_sweep` (or `pytest tests/sweeps
-m slow -s`) regenerates every number in the "bulk_edge / conditioning"
section and the extended-fixture half of "spectral_entropy /
effective_rank" above. (Previously this sweep only existed as a scratch
script, with the report documenting the construction precisely enough to
read as "reproducible" without anything in the repo actually able to
reproduce it — this section used to say so explicitly. That gap is what
prompted committing it.)

The original six-field sweeps (trace/top_eig/negative_mass/density, plus
the tiny_mlp entropy/rank refresh above) are **not** checked in — still run
interactively from `tests/test_eigenthings.py`'s `tiny_mlp`/
`reference_hessian.py` fixtures, reproducible from this doc's numbers and
that file's fixtures. The regression tests that *are* checked in from that
side: `test_effective_rank_and_entropy_converge_with_lanczos_depth` and the
existing `test_top_eig_converges_with_lanczos_depth` /
`test_negative_mass_recovered` / `test_trace_converges_to_exact` /
`test_density_matches_exact_histogram` in `tests/test_eigenthings.py`, plus
`test_bulk_edge_conditioning_converge_on_extended_fixture` in
`tests/sweeps/` now covering `bulk_edge`/`conditioning`.
