# training/spectral_schedule.py

from __future__ import annotations

from collections import deque


class SpectralSchedule:
    """
    Decides which steps get an (expensive) Hessian spectral snapshot.

    Default cadence is geometric (each logged step is `growth`x further out
    than the last), so early training is sampled coarsely without wasting
    budget once steps get large. Whenever the train/test accuracy gap is
    changing fast -- the grokking transition -- it switches to a dense fixed
    stride for as long as the fast change continues, then reverts to
    geometric spacing once things settle again. This only ever gets asked
    about steps that already have a fresh eval (train/test accuracy), since
    the transition-detection signal needs both.
    """

    def __init__(
            self,
            growth: float = 1.15,
            dense_stride: int = 100,
            gap_rate_threshold: float = 1e-4,
    ):
        if growth <= 1.0:
            raise ValueError("growth must be > 1.0 for a geometric schedule")

        self.growth = growth
        self.dense_stride = dense_stride
        self.gap_rate_threshold = gap_rate_threshold

        self._next_geometric_step = 1.0
        self._recent_gaps: deque[tuple[int, float]] = deque(maxlen=2)
        self._last_dense_step: int | None = None

    def update_gap(self, step: int, train_acc: float, test_acc: float) -> None:
        self._recent_gaps.append((step, train_acc - test_acc))

    def _in_transition(self) -> bool:
        if len(self._recent_gaps) < 2:
            return False
        (s0, g0), (s1, g1) = self._recent_gaps[0], self._recent_gaps[-1]
        if s1 <= s0:
            return False
        rate = abs(g1 - g0) / (s1 - s0)
        return rate >= self.gap_rate_threshold

    def should_log(self, step: int) -> bool:
        if step <= 1:
            self._next_geometric_step = max(self._next_geometric_step, step + 1)
            return True

        # Advance the geometric counter whenever it comes due, regardless of
        # whether a dense transition phase is also active right now -- this
        # keeps it from falling far behind `step` during a long dense phase,
        # which would otherwise cause one stale catch-up log right when
        # reverting back to geometric spacing.
        geometric_due = step >= self._next_geometric_step
        if geometric_due:
            self._next_geometric_step = max(step + 1, self._next_geometric_step * self.growth)

        if self._in_transition():
            due = self._last_dense_step is None or (step - self._last_dense_step) >= self.dense_stride
            if due:
                self._last_dense_step = step
                return True
            return False

        return geometric_due
