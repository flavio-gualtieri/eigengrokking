# io_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _as_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _plot_if_valid(ax, x, y, **kwargs) -> None:
    x_np, y_np = _as_numpy(x), _as_numpy(y)
    if x_np is not None and y_np is not None and len(x_np) and len(x_np) == len(y_np):
        ax.plot(x_np, y_np, **kwargs)


def _build_title(cfg) -> str:
    train_info = (
        f"{cfg.train_points:.2f} frac"
        if isinstance(cfg.train_points, float)
        else f"{cfg.train_points} pts"
    )
    focus_info = (
        f"dataset={cfg.dataset}, "
        f"alpha={cfg.initialization_scale}, "
        f"WD={cfg.weight_decay}"
    )
    return (
        f"{focus_info}\n"
        f"depth={cfg.depth}, width={cfg.width}, act={cfg.activation}, "
        f"train={train_info}, bs={cfg.batch_size}, steps={cfg.optimization_steps}"
    )


def _plot_training_with_right_axis(
    path: Path,
    cfg,
    history: Dict[str, Any],
    *,
    right_steps: Sequence,
    right_values: Sequence,
    right_ylabel: str,
    right_legend_label: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    train_steps = history.get("train_steps", [])
    test_steps = history.get("test_steps", [])

    train_accuracies = history.get("train_accuracies", [])
    test_accuracies = history.get("test_accuracies", [])
    adv_test_accuracies = history.get("adv_test_accuracies", [])

    fig, ax = plt.subplots(figsize=(8, 6))

    _plot_if_valid(ax, train_steps, train_accuracies, label="train", color="blue")
    _plot_if_valid(ax, test_steps, test_accuracies, label="test", color="red")

    if len(adv_test_accuracies) == len(test_steps):
        _plot_if_valid(
            ax,
            test_steps,
            adv_test_accuracies,
            label=f"adv test (FGSM ε={cfg.fgsm_epsilon})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Accuracy")

    ax2 = ax.twinx()
    _plot_if_valid(
        ax2,
        right_steps,
        right_values,
        color="purple",
        linestyle="-",
        label=right_legend_label,
    )
    ax2.set_ylabel(right_ylabel)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles1 or handles2:
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    ax.set_title(_build_title(cfg), fontsize=10)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_plot_png(path: Path, cfg, history: Dict[str, Any]) -> None:
    """
    Save the original accuracy plot plus right-axis variants.
    Supports both spectral-density metrics and toy-model Hessian metrics.
    """
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stem, suffix = path.stem, path.suffix
    out_dir = path.parent

    variants = [
        # Spectral-density-derived quantities
        ("_mass_gt1", "eig_mass_gt1", "eig_log_steps", "Mass λ > 1", "Mass λ > 1"),
        ("_phi_1_moment", "phi_1_moment", "eig_log_steps", "Phi 1st Moment", "Phi 1st Moment"),
        ("_phi_2_moment", "phi_2_moment", "eig_log_steps", "Phi 2nd Moment", "Phi 2nd Moment"),
        ("_phi_pos_neg_ratio", "phi_pos_neg_ratio", "eig_log_steps", "Phi Pos-Neg Ratio", "Phi Pos-Neg Ratio"),

        # Weight norms
        ("_weight_norms", "norms", "train_steps", "Weight Norms", "Weight Norms"),
        ("_last_layer_weight_norms", "last_layer_norms", "train_steps", "Last Layer Weight Norms", "Last Layer Weight Norms"),

        # Toy-model full Hessian summaries
        ("_hessian_min_eig", "TOY_hessian_min_eig", "TOY_hessian_steps", "Min Hessian Eigenvalue", "Min Hessian Eigenvalue"),
        ("_hessian_max_eig", "TOY_hessian_max_eig", "TOY_hessian_steps", "Max Hessian Eigenvalue", "Max Hessian Eigenvalue"),
        ("_hessian_trace", "TOY_hessian_trace", "TOY_hessian_steps", "Hessian Trace", "Hessian Trace"),
        ("_hessian_spectral_radius", "TOY_hessian_spectral_radius", "TOY_hessian_steps", "Hessian Spectral Radius", "Hessian Spectral Radius"),
    ]

    for suffix_part, history_key, step_key, ylabel, legend in variants:
        values = history.get(history_key, [])
        steps = history.get(step_key, [])

        if not values or not steps or len(values) != len(steps):
            continue
        
        _plot_training_with_right_axis(
            out_dir / f"{stem}{suffix_part}{suffix}",
            cfg,
            history,
            right_steps=history.get(step_key, []),
            right_values=history.get(history_key, []),
            right_ylabel=ylabel,
            right_legend_label=legend,
        )


def _resolve_x_limits(
    t_store: Dict[int, torch.Tensor],
) -> tuple[Optional[tuple[float, float]], Dict[int, np.ndarray]]:
    t_store_np = {step: _as_numpy(v) for step, v in t_store.items()}
    valid = [arr for arr in t_store_np.values() if arr is not None and arr.size > 0]

    if not valid:
        raise ValueError("t_store must contain at least one non-empty tensor.")

    xlim = (
        min(float(arr[0]) for arr in valid),
        max(float(arr[-1]) for arr in valid),
    )
    return xlim, t_store_np


def save_log_spectral_snapshots(
    *,
    spectra_path: Path,
    phi_store: Dict[int, torch.Tensor],
    t_store: Dict[int, torch.Tensor],
    dpi: int = 200,
    eps: float = 1e-12,
) -> None:
    """
    Saves per-step snapshots of log spectral density:
        x-axis: eigenvalue λ
        y-axis: log φ(λ)

    Uses only t_store for x-values.
    """
    spectra_path.mkdir(parents=True, exist_ok=True)

    steps = sorted(phi_store)
    if not steps:
        return

    xlim, t_store_np = _resolve_x_limits(t_store)
    phi_store_np = {step: _as_numpy(phi_store[step]) for step in steps}

    phi0_log = np.log(np.clip(phi_store_np[steps[0]], eps, None))
    y_min = float(np.nanmin(phi0_log))
    y_max = float(np.nanmax(phi0_log))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_lim = (y_min - pad, y_max + pad)

    for step in steps:
        phi_log = np.log(np.clip(phi_store_np[step], eps, None))
        t_np = t_store_np.get(step)

        if t_np is None:
            raise ValueError(f"Missing t_store entry for step {step}.")

        n = min(len(t_np), len(phi_log))
        if n == 0:
            continue

        fig, ax = plt.subplots()
        ax.plot(t_np[:n], phi_log[:n])
        ax.set_xlabel("Eigenvalue [λ]")
        ax.set_ylabel("log Density [log φ(λ)]")
        ax.set_title(f"Log spectral density at step {step}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*y_lim)

        fig.tight_layout()
        fig.savefig(spectra_path / f"log_spectrum_step_{step}.png", dpi=dpi)
        plt.close(fig)