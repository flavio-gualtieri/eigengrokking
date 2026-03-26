# io_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch


def _plotly_counterpart_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_PLOTLY.html")


def _as_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _plot_if_valid(ax, x, y, **kwargs) -> None:
    x_np = _as_numpy(x)
    y_np = _as_numpy(y)
    if x_np is None or y_np is None:
        return
    if len(x_np) == 0 or len(x_np) != len(y_np):
        return
    ax.plot(x_np, y_np, **kwargs)


def _build_title(cfg) -> str:
    train_info = (
        f"{cfg.train_points:.2f} frac"
        if isinstance(cfg.train_points, float)
        else f"{cfg.train_points} pts"
    )
    return (
        f"dataset={cfg.dataset}, alpha={cfg.initialization_scale}, WD={cfg.weight_decay}\n"
        f"depth={cfg.depth}, width={cfg.width}, act={cfg.activation}, "
        f"train={train_info}, bs={cfg.batch_size}, steps={cfg.optimization_steps}"
    )


def _plot_training_with_right_axis(
    path: Path,
    cfg,
    history: dict[str, Any],
    *,
    right_steps: Sequence,
    right_values: Sequence,
    right_ylabel: str,
    right_legend_label: str,
    log_x: bool,
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

    if log_x:
        ax.set_xscale("log")
        ax.set_xlabel("Optimization Steps (LOG)")
    else:
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


def _plot_training_with_right_axis_plotly(
    path: Path,
    cfg,
    history: dict[str, Any],
    *,
    right_steps: Sequence,
    right_values: Sequence,
    right_ylabel: str,
    right_legend_label: str,
    log_x: bool,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    train_steps = history.get("train_steps", [])
    test_steps = history.get("test_steps", [])
    train_accuracies = history.get("train_accuracies", [])
    test_accuracies = history.get("test_accuracies", [])
    adv_test_accuracies = history.get("adv_test_accuracies", [])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    x_np = _as_numpy(train_steps)
    y_np = _as_numpy(train_accuracies)
    if x_np is not None and y_np is not None and len(x_np) == len(y_np) and len(x_np) > 0:
        fig.add_trace(
            go.Scatter(x=x_np, y=y_np, mode="lines", name="train"),
            secondary_y=False,
        )

    x_np = _as_numpy(test_steps)
    y_np = _as_numpy(test_accuracies)
    if x_np is not None and y_np is not None and len(x_np) == len(y_np) and len(x_np) > 0:
        fig.add_trace(
            go.Scatter(x=x_np, y=y_np, mode="lines", name="test"),
            secondary_y=False,
        )

    x_np = _as_numpy(test_steps)
    y_np = _as_numpy(adv_test_accuracies)
    if x_np is not None and y_np is not None and len(x_np) == len(y_np) and len(x_np) > 0:
        fig.add_trace(
            go.Scatter(
                x=x_np,
                y=y_np,
                mode="lines",
                name=f"adv test (FGSM ε={cfg.fgsm_epsilon})",
            ),
            secondary_y=False,
        )

    x_np = _as_numpy(right_steps)
    y_np = _as_numpy(right_values)
    if x_np is not None and y_np is not None and len(x_np) == len(y_np) and len(x_np) > 0:
        fig.add_trace(
            go.Scatter(x=x_np, y=y_np, mode="lines", name=right_legend_label),
            secondary_y=True,
        )

    fig.update_xaxes(
        title_text="Optimization Steps (LOG)" if log_x else "Optimization Steps",
        type="log" if log_x else "linear",
    )
    fig.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig.update_yaxes(title_text=right_ylabel, secondary_y=True)

    fig.update_layout(
        title=_build_title(cfg),
        width=800,
        height=600,
    )

    if path.suffix.lower() == ".html":
        fig.write_html(path)
    else:
        fig.write_image(path)


PLOT_VARIANTS = [
    ("_mass_gt1", "eig_mass_gt1", "eig_steps", "Mass λ > 1"),
    ("_phi_1_moment", "phi_1_moment", "eig_steps", "Phi 1st Moment"),
    ("_phi_2_moment", "phi_2_moment", "eig_steps", "Phi 2nd Moment"),
    ("_phi_pos_neg_ratio", "phi_pos_neg_ratio", "eig_steps", "Phi Pos-Neg Ratio"),
    ("_top_eig", "top_eig", "eig_steps", "Top Eigenvalue"),
    ("_trace", "trace", "eig_steps", "Hessian Trace"),

    ("_weight_norms", "norms", "train_steps", "Weight Norms"),
    ("_last_layer_weight_norms", "last_layer_norms", "train_steps", "Last Layer Weight Norms"),

    ("_hessian_min_eig", "TOY_hessian_min_eig", "TOY_hessian_steps", "Min Hessian Eigenvalue"),
    ("_hessian_max_eig", "TOY_hessian_max_eig", "TOY_hessian_steps", "Max Hessian Eigenvalue"),
    ("_hessian_trace", "TOY_hessian_trace", "TOY_hessian_steps", "Hessian Trace"),
    ("_hessian_spectral_radius", "TOY_hessian_spectral_radius", "TOY_hessian_steps", "Hessian Spectral Radius"),
]

def save_training_plot(
    path: Path,
    cfg,
    history: dict[str, Any],
    *,
    log_x: bool = True,
) -> None:
    """
    Save accuracy plots with additional right-axis metrics.
    """
    path = Path(path)
    if not log_x:
        path = path.parent / "no_log" / path.name

    path.parent.mkdir(parents=True, exist_ok=True)

    stem = path.stem
    suffix = path.suffix
    out_dir = path.parent
    suffix_extra = "" if log_x else "_nonlog"

    for suffix_part, history_key, step_key, ylabel in PLOT_VARIANTS:
        values = history.get(history_key, [])
        steps = history.get(step_key, [])

        if not values or not steps or len(values) != len(steps):
            continue

        mpl_path = out_dir / f"{stem}{suffix_part}{suffix_extra}{suffix}"

        _plot_training_with_right_axis(
            mpl_path,
            cfg,
            history,
            right_steps=steps,
            right_values=values,
            right_ylabel=ylabel,
            right_legend_label=ylabel,
            log_x=log_x,
        )

        _plot_training_with_right_axis_plotly(
            _plotly_counterpart_path(mpl_path),
            cfg,
            history,
            right_steps=steps,
            right_values=values,
            right_ylabel=ylabel,
            right_legend_label=ylabel,
            log_x=log_x,
        )


def _resolve_x_limits(
    t_store: dict[int, torch.Tensor],
) -> tuple[tuple[float, float], dict[int, np.ndarray]]:
    t_store_np = {step: _as_numpy(v) for step, v in t_store.items()}
    valid = [arr for arr in t_store_np.values() if arr is not None and arr.size > 0]

    if not valid:
        raise ValueError("t_store must contain at least one non-empty tensor.")

    return (
        min(float(arr[0]) for arr in valid),
        max(float(arr[-1]) for arr in valid),
    ), t_store_np


def _save_log_spectral_snapshot_plotly(
    path: Path,
    *,
    step: int,
    t_np: np.ndarray,
    phi_log: np.ndarray,
    xlim: tuple[float, float],
    y_lim: tuple[float, float],
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t_np, y=phi_log, mode="lines", name=f"step {step}")
    )
    fig.update_layout(
        title=f"Log spectral density at step {step}",
        xaxis_title="Eigenvalue [λ]",
        yaxis_title="log Density [log φ(λ)]",
        width=700,
        height=500,
    )
    fig.update_xaxes(range=[xlim[0], xlim[1]])
    fig.update_yaxes(range=[y_lim[0], y_lim[1]])

    if path.suffix.lower() == ".html":
        fig.write_html(path)
    else:
        fig.write_image(path)


def save_log_spectral_snapshots(
    *,
    spectra_path: Path,
    phi_store: dict[int, torch.Tensor],
    t_store: dict[int, torch.Tensor],
    dpi: int = 200,
    eps: float = 1e-12,
) -> None:
    """
    Save per-step snapshots of log spectral density.
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
        t_np = t_store_np.get(step)
        if t_np is None:
            raise ValueError(f"Missing t_store entry for step {step}.")

        phi_log = np.log(np.clip(phi_store_np[step], eps, None))
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

        mpl_path = spectra_path / f"log_spectrum_step_{step}.png"
        fig.savefig(mpl_path, dpi=dpi)
        plt.close(fig)

        _save_log_spectral_snapshot_plotly(
            _plotly_counterpart_path(mpl_path),
            step=step,
            t_np=t_np[:n],
            phi_log=phi_log[:n],
            xlim=xlim,
            y_lim=y_lim,
        )