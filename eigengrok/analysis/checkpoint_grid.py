# eigengrok/analysis/checkpoint_grid.py

"""
Which checkpoints of a run Phase 2 computes Hessian spectra at, and so which
ones the sweep rerun keeps (scripts/prune_checkpoints.py):

  - a log-spaced grid over the whole run (PER_DECADE points per decade of
    steps, each snapped to the nearest saved checkpoint), and
  - every saved checkpoint in [WINDOW_LO, WINDOW_HI] x t_onset, the run's own
    onset (eigengrok/analysis/milestones.py), every 100 steps in the sweep.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

PER_DECADE = 10
WINDOW_LO, WINDOW_HI = 0.5, 1.5


def log_grid(available: Iterable[int], *, per_decade: int = PER_DECADE) -> list[int]:
    """Saved steps nearest (in log-step) to a geometric grid from the first to the last positive one."""
    steps = np.array(sorted({int(s) for s in available if s > 0}))
    if steps.size == 0:
        return []
    lo, hi = math.log10(steps[0]), math.log10(steps[-1])
    targets = np.logspace(lo, hi, max(int(math.ceil((hi - lo) * per_decade)) + 1, 2))
    nearest = np.abs(np.log10(steps)[None, :] - np.log10(targets)[:, None]).argmin(axis=1)
    return sorted({int(steps[i]) for i in nearest})


def onset_window(available: Iterable[int], t_onset: float, *, lo: float = WINDOW_LO, hi: float = WINDOW_HI) -> list[int]:
    """Every saved step in [lo * t_onset, hi * t_onset]; none if t_onset is NaN."""
    if not math.isfinite(t_onset):
        return []
    return sorted(int(s) for s in set(available) if lo * t_onset <= s <= hi * t_onset)


def phase2_steps(available: Iterable[int], t_onset: float) -> list[int]:
    available = list(available)
    return sorted(set(log_grid(available)) | set(onset_window(available, t_onset)))
