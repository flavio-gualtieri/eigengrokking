# eigengrok/train/spectral_schedule.py

from __future__ import annotations


def spectral_steps(total_steps: int, *, growth: float, every: int) -> list[int]:
    """
    Steps at which a run takes a Hessian snapshot: a geometric grid (each step
    `growth` times the previous, at least one apart) from step 1 up to
    `every`, to resolve early training, then every `every` steps.

    The schedule depends on the config alone. An earlier version switched to
    dense sampling whenever the train/test accuracy gap moved fast. That made
    snapshot times depend on test accuracy, which leaks the label a grokking
    forecaster is trying to predict into its input features.
    """
    if growth <= 1.0:
        raise ValueError("growth must be > 1.0 for a geometric schedule")
    if every < 1:
        raise ValueError("every must be >= 1")

    steps = {0, 1}
    s = 1.0
    while s < every:
        steps.add(int(round(s)))
        s = max(s * growth, s + 1)
    steps.update(range(0, total_steps, every))
    return sorted(step for step in steps if step < total_steps)
