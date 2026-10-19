# eigengrok/analysis/changepoints.py

"""
When does each indicator change around the grokking transition? The test
behind the claim that the effective-rank collapse happens *at* the transition
rather than before it.

Per run and per series (an indicator, or test loss as the reference), on the
run's own checkpoint steps within the window [WINDOW_LO, WINDOW_HI] x t_onset
(the checkpoints kept every 100 steps around each run's onset,
eigengrok/analysis/checkpoint_grid.py):

  1. transform the values as the forecast features do (log, log1p or asinh;
     eigengrok/analysis/forecast_features.py) and remove spikes with a
     Hampel filter: a point more than HAMPEL_SIGMAS noise SDs from its
     DESPIKE-point running median is replaced by that median, the noise SD
     estimated robustly from second differences so a trend does not count
     as noise. That removes spikes up to two checkpoints wide (SLQ outliers,
     which can repeat at neighbouring checkpoints because a run reuses its
     probes at every checkpoint; train-loss blips) and keeps any change that
     lasts 3+ checkpoints. Every other point stays as measured: smoothing
     them all would correlate neighbouring residuals, and BIC (step 3)
     assumes they are independent (with a 5-point running median, a straight
     line plus white noise "changes" in ~95% of draws);
  2. segment the series into 1..MAX_SEGMENTS linear pieces (each its own
     line, at least MIN_SEGMENT points) by exact least squares (dynamic
     programming over every split), and pick the number of pieces by BIC;
  3. the series changes if the selected segmentation beats a single line by
     at least MIN_DELTA_BIC. Its changepoint is the start of its STEEPEST
     EPISODE: take the piece with the largest net slope (fitted end minus
     the previous piece's fitted end, over that span, so a jump into the
     piece counts), then extend backwards over adjacent pieces moving the
     same way at least STEEP_FRACTION as fast. The episode's first step is
     the changepoint. For test loss that is where it leaves its
     post-memorization plateau (a drop split into a steep part and a tail is
     still one episode, a slowly declining plateau is not part of it); for an
     effective-rank collapse it is where the collapse begins.

If the episode starts with the first piece, the steepest change is already
under way when the window opens: the changepoint is the window start, flagged
`at_window_start`, and its lead time is a lower bound.

The window is the dense one, not [t_mem, ...]: right after memorization most
indicators move faster per step than at any later point, so over a window
starting at t_mem the steepest episode is that transient, not anything
related to the transition. Measuring time in log-step would hide the
transient but make every late change look steeper, a bias towards finding
changes at the transition, which is the claim under test.

The transition start is test loss's changepoint, dating its steepest FALL
(a post-grokking collapse late in the window can make a rise steeper).
Indicators are dated by their steepest change either way, unless
`direction` asks for one. Lead time = transition start - indicator
changepoint: positive means the indicator changes first. Every series of a
run is read at the same steps, so no indicator gets finer time resolution
than another (100 steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eigengrok.analysis.checkpoint_grid import WINDOW_HI, WINDOW_LO
from eigengrok.analysis.forecast_features import B1_INDICATORS, C_INDICATORS, RUN, _transform

DESPIKE = 5          # running-median width of the spike filter, in checkpoints
HAMPEL_SIGMAS = 3.0  # a point this many robust SDs from its running median is a spike
MAX_SEGMENTS = 5
MIN_SEGMENT = 3      # points per linear piece
# BIC improvement over a single line needed to call a change ("very strong" evidence). Checked by simulation: a
# straight line plus white or heavy-tailed (Student t, 3 dof) noise, spike-filtered like the data, clears it in at most
# ~3% of draws at 30-150 points. Errors correlated between neighbouring checkpoints would clear it more often, so the
# report also runs a stricter margin.
MIN_DELTA_BIC = 10
STEEP_FRACTION = 1 / 3  # an adjacent piece joins the steepest episode if it moves the same way at least this fast

REFERENCE = "test_loss"
REFERENCE_DIRECTION = -1  # the transition start is where test loss starts to FALL off its plateau
INDICATORS = (*C_INDICATORS, *B1_INDICATORS)


@dataclass(frozen=True)
class Changepoint:
    step: float             # first step of the steepest episode; NaN if no change was detected
    change: float           # net change of the fitted trend over the episode (transformed units)
    slope: float            # net slope of the steepest piece (transformed units per 1000 steps)
    at_window_start: bool   # the episode starts with the first piece: already under way at the window start
    boundaries: tuple       # first step of every segment after the first, in step order
    delta_bic: float        # BIC(single line) - BIC(selected segmentation)
    n_points: int

    @property
    def detected(self) -> bool:
        return bool(np.isfinite(self.step))


def running_median(v: np.ndarray, width: int = DESPIKE) -> np.ndarray:
    """
    Centred running median of odd `width`, the series mirrored at each end
    (without repeating the end point), so a spike at either end is removed
    like any other.
    """
    v = np.asarray(v, dtype=float)
    half = min(width // 2, v.size - 1)
    if half < 1:
        return v.copy()
    padded = np.pad(v, half, mode="reflect")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)


def noise_scale(v: np.ndarray) -> float:
    """
    Robust SD of the point-to-point noise: 1.4826 x the median absolute
    deviation of the second differences, / sqrt(6) (their SD for white noise).
    Second differences cancel any locally linear trend, and a spike moves
    only three of them.
    """
    d2 = np.diff(np.asarray(v, dtype=float), 2)
    if d2.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6))


def despike(v: np.ndarray, width: int = DESPIKE, sigmas: float = HAMPEL_SIGMAS) -> np.ndarray:
    """
    Hampel filter: points more than `sigmas` x noise_scale(v) from their
    `width`-point running median are replaced by it; the rest are kept.
    """
    v = np.asarray(v, dtype=float)
    med = running_median(v, width)
    scale = noise_scale(v)
    if not scale > 0:
        return v.copy()
    return np.where(np.abs(v - med) > sigmas * scale, med, v)


def _line_rss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """rss[i, j]: residual sum of squares of the least-squares line through points i..j-1 (inf if j <= i)."""
    n = x.size
    cum = {name: np.concatenate([[0.0], np.cumsum(v)]) for name, v in
           (("x", x), ("y", y), ("xx", x * x), ("xy", x * y), ("yy", y * y))}
    rss = np.full((n, n + 1), np.inf)
    for i in range(n):
        j = np.arange(i + 1, n + 1)
        m = j - i
        sx, sy = cum["x"][j] - cum["x"][i], cum["y"][j] - cum["y"][i]
        vxx = cum["xx"][j] - cum["xx"][i] - sx * sx / m
        vxy = cum["xy"][j] - cum["xy"][i] - sx * sy / m
        vyy = cum["yy"][j] - cum["yy"][i] - sy * sy / m
        safe = vxx > 1e-12
        rss[i, i + 1:] = np.maximum(np.where(safe, vyy - vxy ** 2 / np.where(safe, vxx, 1.0), vyy), 0.0)
    return rss


def segmentations(x: np.ndarray, y: np.ndarray, *, max_segments: int = MAX_SEGMENTS,
                  min_segment: int = MIN_SEGMENT) -> dict[int, tuple[float, list[int]]]:
    """
    For every k = 1..max_segments that fits: (rss, starts) of the exact
    least-squares split of (x, y) into k linear pieces of >= min_segment
    points each; `starts` are the pieces' first indices, the first always 0.
    """
    n = x.size
    cost = _line_rss(x, y)
    for i in range(n):
        cost[i, i + 1:min(i + min_segment, n + 1)] = np.inf
    best = np.full((max_segments + 1, n + 1), np.inf)
    back = np.zeros((max_segments + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, max_segments + 1):
        for j in range(1, n + 1):
            total = best[k - 1, :j] + cost[:j, j]
            i = int(np.argmin(total))
            best[k, j], back[k, j] = total[i], i
    out = {}
    for k in range(1, max_segments + 1):
        if np.isfinite(best[k, n]):
            starts, j = [], n
            for kk in range(k, 0, -1):
                j = back[kk, j]
                starts.append(j)
            out[k] = (float(best[k, n]), starts[::-1])
    return out


def _fitted_ends(x: np.ndarray, y: np.ndarray, starts: list[int]) -> list[tuple[float, float]]:
    """(fitted value at the first point, at the last point) of each piece's own line."""
    ends = []
    for a, b in zip(starts, [*starts[1:], x.size]):
        slope, intercept = np.polyfit(x[a:b], y[a:b], 1)
        ends.append((slope * x[a] + intercept, slope * x[b - 1] + intercept))
    return ends


def fit_changepoint(steps, values, *, max_segments: int = MAX_SEGMENTS, min_segment: int = MIN_SEGMENT,
                    min_delta_bic: float = MIN_DELTA_BIC, fraction: float = STEEP_FRACTION,
                    despike_width: int = DESPIKE, direction: int = 0) -> Changepoint:
    """
    The changepoint of one (already transformed) series; see the module
    docstring. `direction` -1 (or +1) dates the steepest fall (rise) only,
    ignoring steeper changes the other way; 0 dates whichever is steepest.
    """
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(steps) & np.isfinite(values)
    order = np.argsort(steps[keep])
    steps, values = steps[keep][order], values[keep][order]
    n = steps.size
    if n < 2 * min_segment:
        return Changepoint(np.nan, np.nan, np.nan, False, (), np.nan, n)

    y = despike(values, despike_width) if despike_width > 1 else values
    x = (steps - steps[0]) / 1000.0
    fits = segmentations(x, y, max_segments=max_segments, min_segment=min_segment)
    # k pieces: k intercepts, k slopes and k - 1 split points
    bic = {k: n * np.log(max(rss, 1e-300) / n) + (3 * k - 1) * np.log(n) for k, (rss, _) in fits.items()}
    k = min(bic, key=bic.get)
    delta = float(bic[1] - bic[k])
    if k == 1 or delta < min_delta_bic:
        return Changepoint(np.nan, np.nan, np.nan, False, (), delta, n)

    starts = fits[k][1]
    ends = _fitted_ends(x, y, starts)
    # net change and span of each piece, from the previous piece's fitted end (so a jump into it counts)
    last = [b - 1 for b in [*starts[1:], n]]
    changes = [ends[0][1] - ends[0][0]] + [ends[s][1] - ends[s - 1][1] for s in range(1, k)]
    spans = [x[last[0]] - x[0]] + [x[last[s]] - x[last[s - 1]] for s in range(1, k)]
    slopes = np.array(changes) / np.array(spans)
    steepness = np.abs(slopes)
    if direction:
        steepness = np.where(np.sign(slopes) == np.sign(direction), steepness, -1.0)
    if steepness.max() <= 0:  # nothing moves the requested way
        return Changepoint(np.nan, np.nan, np.nan, False, (), delta, n)
    top = int(np.argmax(steepness))
    first = top
    while first > 0 and np.sign(slopes[first - 1]) == np.sign(slopes[top]) \
            and abs(slopes[first - 1]) >= fraction * abs(slopes[top]):
        first -= 1
    return Changepoint(step=float(steps[starts[first]]), change=float(sum(changes[first:top + 1])),
                       slope=float(slopes[top]), at_window_start=first == 0,
                       boundaries=tuple(float(steps[a]) for a in starts[1:]), delta_bic=delta, n_points=n)


def run_series(curves: pd.DataFrame, spectra: pd.DataFrame, t_onset: float,
               *, window: tuple[float, float] = (WINDOW_LO, WINDOW_HI)) -> pd.DataFrame:
    """
    One run's reference and indicator series on its spectral checkpoint steps
    within `window` x t_onset, transformed: columns step, test_loss and every
    INDICATORS column present.
    """
    cheap = [c for c in (REFERENCE, *B1_INDICATORS) if c in curves]
    df = spectra[["step", *[c for c in C_INDICATORS if c in spectra]]].merge(
        curves[["step", *cheap]], on="step", how="inner", validate="one_to_one")
    df = df[(df["step"] >= window[0] * t_onset) & (df["step"] <= window[1] * t_onset)].sort_values("step").reset_index(drop=True)
    for c in df.columns.drop("step"):
        df[c] = _transform(c, df[c].to_numpy(dtype=float))
    return df


def changepoint_table(curves: pd.DataFrame, spectra: pd.DataFrame, milestones: pd.DataFrame,
                      *, window: tuple[float, float] = (WINDOW_LO, WINDOW_HI), direction: int = 0,
                      **fit_kwargs) -> pd.DataFrame:
    """
    One row per (run, series) with the changepoint of the reference (series
    == REFERENCE) and of every indicator, the run's t_mem and t_onset, its
    transition start `ref_step` (the reference's changepoint), and the lead
    time: lead = ref_step - changepoint in steps (> 0: the indicator changes
    first) and lead_frac = lead / t_onset. `direction` applies to the
    indicators; the reference always uses REFERENCE_DIRECTION.
    """
    curves_by_run = dict(tuple(curves.groupby(RUN, sort=False)))
    spectra_by_run = dict(tuple(spectra.groupby(RUN, sort=False)))
    rows = []
    for _, m in milestones.iterrows():
        key = tuple(m[k] for k in RUN)
        if key not in curves_by_run or key not in spectra_by_run:
            continue
        df = run_series(curves_by_run[key], spectra_by_run[key], m["t_onset"], window=window)
        for series in df.columns.drop("step"):
            sign = REFERENCE_DIRECTION if series == REFERENCE else direction
            cp = fit_changepoint(df["step"], df[series], direction=sign, **fit_kwargs)
            rows.append({**dict(zip(RUN, key)), "t_mem": m["t_mem"], "t_onset": m["t_onset"], "series": series,
                         "changepoint": cp.step, "change": cp.change, "slope": cp.slope,
                         "at_window_start": cp.at_window_start, "delta_bic": cp.delta_bic, "n_points": cp.n_points,
                         "boundaries": " ".join(f"{b:g}" for b in cp.boundaries),
                         "window_start": df["step"].iloc[0] if len(df) else np.nan})
    table = pd.DataFrame(rows)
    ref = table.loc[table["series"] == REFERENCE, RUN + ["changepoint"]].rename(columns={"changepoint": "ref_step"})
    table = table.merge(ref, on=RUN, how="left", validate="many_to_one")
    table["lead"] = table["ref_step"] - table["changepoint"]
    table["lead_frac"] = table["lead"] / table["t_onset"]
    return table
