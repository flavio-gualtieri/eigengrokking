# io_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

def _add_plotly_prefix(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f"PLOTLY_{path.name}")


def _as_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _is_valid_xy(x: Any, y: Any) -> bool:
    x_np, y_np = _as_numpy(x), _as_numpy(y)
    return (
        x_np is not None
        and y_np is not None
        and len(x_np) > 0
        and len(x_np) == len(y_np)
    )


def _add_trace_if_valid(
    fig: go.Figure,
    x: Any,
    y: Any,
    *,
    name: str,
    row: int = 1,
    col: int = 1,
    secondary_y: bool = False,
    **kwargs,
) -> None:
    x_np, y_np = _as_numpy(x), _as_numpy(y)
    if _is_valid_xy(x_np, y_np):
        fig.add_trace(
            go.Scatter(
                x=x_np,
                y=y_np,
                mode="lines",
                name=name,
                **kwargs,
            ),
            row=row,
            col=col,
            secondary_y=secondary_y,
        )


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
        f"{focus_info}<br>"
        f"depth={cfg.depth}, width={cfg.width}, act={cfg.activation}, "
        f"train={train_info}, bs={cfg.batch_size}, steps={cfg.optimization_steps}"
    )


def _normalize_html_path(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".html":
        path = path.with_suffix(".html")
    return path


def _plot_training_with_right_axis_log(
    path: Path,
    cfg,
    history: Dict[str, Any],
    *,
    right_steps: Sequence,
    right_values: Sequence,
    right_ylabel: str,
    right_legend_label: str,
) -> None:
    path = _add_plotly_prefix(_normalize_html_path(path))

    train_steps = history.get("train_steps", [])
    test_steps = history.get("test_steps", [])

    train_accuracies = history.get("train_accuracies", [])
    test_accuracies = history.get("test_accuracies", [])
    adv_test_accuracies = history.get("adv_test_accuracies", [])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    _add_trace_if_valid(
        fig,
        train_steps,
        train_accuracies,
        name="train",
        line=dict(color="blue"),
    )
    _add_trace_if_valid(
        fig,
        test_steps,
        test_accuracies,
        name="test",
        line=dict(color="red"),
    )

    if len(adv_test_accuracies) == len(test_steps):
        _add_trace_if_valid(
            fig,
            test_steps,
            adv_test_accuracies,
            name=f"adv test (FGSM ε={cfg.fgsm_epsilon})",
        )

    _add_trace_if_valid(
        fig,
        right_steps,
        right_values,
        name=right_legend_label,
        secondary_y=True,
        line=dict(color="purple"),
    )

    fig.update_xaxes(
        title_text="Optimization Steps (LOG)",
        type="log",
    )
    fig.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig.update_yaxes(title_text=right_ylabel, secondary_y=True)

    fig.update_layout(
        title=_build_title(cfg),
        template="plotly_white",
        width=900,
        height=600,
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
        margin=dict(l=60, r=60, t=100, b=60),
        hovermode="x unified",
    )

    fig.write_html(path, include_plotlyjs="cdn")


def _plot_training_with_right_axis_reg(
    path: Path,
    cfg,
    history: Dict[str, Any],
    *,
    right_steps: Sequence,
    right_values: Sequence,
    right_ylabel: str,
    right_legend_label: str,
) -> None:
    path = _add_plotly_prefix(_normalize_html_path(path)) / 'non_log'

    train_steps = history.get("train_steps", [])
    test_steps = history.get("test_steps", [])

    train_accuracies = history.get("train_accuracies", [])
    test_accuracies = history.get("test_accuracies", [])
    adv_test_accuracies = history.get("adv_test_accuracies", [])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    _add_trace_if_valid(
        fig,
        train_steps,
        train_accuracies,
        name="train",
        line=dict(color="blue"),
    )
    _add_trace_if_valid(
        fig,
        test_steps,
        test_accuracies,
        name="test",
        line=dict(color="red"),
    )

    if len(adv_test_accuracies) == len(test_steps):
        _add_trace_if_valid(
            fig,
            test_steps,
            adv_test_accuracies,
            name=f"adv test (FGSM ε={cfg.fgsm_epsilon})",
        )

    _add_trace_if_valid(
        fig,
        right_steps,
        right_values,
        name=right_legend_label,
        secondary_y=True,
        line=dict(color="purple"),
    )

    fig.update_xaxes(
        title_text="Optimization Steps (LOG)",
    )
    fig.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig.update_yaxes(title_text=right_ylabel, secondary_y=True)

    fig.update_layout(
        title=_build_title(cfg),
        template="plotly_white",
        width=900,
        height=600,
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
        margin=dict(l=60, r=60, t=100, b=60),
        hovermode="x unified",
    )

    fig.write_html(path, include_plotlyjs="cdn")


def save_training_plot_html_log(path: Path, cfg, history: Dict[str, Any]) -> None:
    """
    Save interactive accuracy plots plus right-axis variants as HTML.
    Supports both spectral-density metrics and toy-model Hessian metrics.
    """
    path = _normalize_html_path(path)

    stem = path.stem
    out_dir = path.parent

    variants = [
        # Spectral-density-derived quantities
        ("_mass_gt1_nonlog", "eig_mass_gt1", "eig_steps", "Mass λ > 1", "Mass λ > 1"),
        ("_phi_1_moment_nonlog", "phi_1_moment", "eig_steps", "Phi 1st Moment", "Phi 1st Moment"),
        ("_phi_2_moment_nonlog", "phi_2_moment", "eig_steps", "Phi 2nd Moment", "Phi 2nd Moment"),
        ("_phi_pos_neg_ratio_nonlog", "phi_pos_neg_ratio", "eig_steps", "Phi Pos-Neg Ratio", "Phi Pos-Neg Ratio"),
        ("_top_eig", "top_eig", "eig_steps", "Top Eigenvalue", "Top Eigenvalue"),
        ("_trace", "trace", "eig_steps", "Hessian Trace", "Hessian Trace"),

        # Weight norms
        ("_weight_norms_nonlog", "norms", "train_steps", "Weight Norms", "Weight Norms"),
        ("_last_layer_weight_norms_nonlog", "last_layer_norms", "train_steps", "Last Layer Weight Norms", "Last Layer Weight Norms"),

        # Toy-model full Hessian summaries
        ("_hessian_min_eig_nonlog", "TOY_hessian_min_eig", "TOY_hessian_steps", "Min Hessian Eigenvalue", "Min Hessian Eigenvalue"),
        ("_hessian_max_eig_nonlog", "TOY_hessian_max_eig", "TOY_hessian_steps", "Max Hessian Eigenvalue", "Max Hessian Eigenvalue"),
        ("_hessian_trace_nonlog", "TOY_hessian_trace", "TOY_hessian_steps", "Hessian Trace", "Hessian Trace"),
        ("_hessian_spectral_radius_nonlog", "TOY_hessian_spectral_radius", "TOY_hessian_steps", "Hessian Spectral Radius", "Hessian Spectral Radius"),
    ]

    for suffix_part, history_key, step_key, ylabel, legend in variants:
        values = history.get(history_key, [])
        steps = history.get(step_key, [])

        if not values or not steps or len(values) != len(steps):
            continue

        _plot_training_with_right_axis_log(
            out_dir / f"{stem}{suffix_part}.html",
            cfg,
            history,
            right_steps=steps,
            right_values=values,
            right_ylabel=ylabel,
            right_legend_label=legend,
        )


def save_training_plot_html_reg(path: Path, cfg, history: Dict[str, Any]) -> None:
    """
    Save interactive accuracy plots plus right-axis variants as HTML.
    Supports both spectral-density metrics and toy-model Hessian metrics.
    """
    path = _normalize_html_path(path)  / 'non_log'

    stem = path.stem
    out_dir = path.parent

    variants = [
        # Spectral-density-derived quantities
        ("_mass_gt1_nonlog", "eig_mass_gt1", "eig_log_steps", "Mass λ > 1", "Mass λ > 1"),
        ("_phi_1_moment_nonlog", "phi_1_moment", "eig_log_steps", "Phi 1st Moment", "Phi 1st Moment"),
        ("_phi_2_moment_nonlog", "phi_2_moment", "eig_log_steps", "Phi 2nd Moment", "Phi 2nd Moment"),
        ("_phi_pos_neg_ratio_nonlog", "phi_pos_neg_ratio", "eig_log_steps", "Phi Pos-Neg Ratio", "Phi Pos-Neg Ratio"),

        # Weight norms
        ("_weight_norms_nonlog", "norms", "train_steps", "Weight Norms", "Weight Norms"),
        ("_last_layer_weight_norms_nonlog", "last_layer_norms", "train_steps", "Last Layer Weight Norms", "Last Layer Weight Norms"),

        # Toy-model full Hessian summaries
        ("_hessian_min_eig_nonlog", "TOY_hessian_min_eig", "TOY_hessian_steps", "Min Hessian Eigenvalue", "Min Hessian Eigenvalue"),
        ("_hessian_max_eig_nonlog", "TOY_hessian_max_eig", "TOY_hessian_steps", "Max Hessian Eigenvalue", "Max Hessian Eigenvalue"),
        ("_hessian_trace_nonlog", "TOY_hessian_trace", "TOY_hessian_steps", "Hessian Trace", "Hessian Trace"),
        ("_hessian_spectral_radius_nonlog", "TOY_hessian_spectral_radius", "TOY_hessian_steps", "Hessian Spectral Radius", "Hessian Spectral Radius"),
    ]

    for suffix_part, history_key, step_key, ylabel, legend in variants:
        values = history.get(history_key, [])
        steps = history.get(step_key, [])

        if not values or not steps or len(values) != len(steps):
            continue

        _plot_training_with_right_axis_reg(
            out_dir / f"{stem}{suffix_part}.html",
            cfg,
            history,
            right_steps=steps,
            right_values=values,
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


def save_log_spectral_snapshots_html(
    *,
    spectra_path: Path,
    phi_store: Dict[int, torch.Tensor],
    t_store: Dict[int, torch.Tensor],
    eps: float = 1e-12,
) -> None:
    """
    Saves per-step snapshots of log spectral density as interactive HTML:
        x-axis: eigenvalue λ
        y-axis: log φ(λ)

    Uses only t_store for x-values.
    """
    spectra_path = Path(spectra_path)
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

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t_np[:n],
                y=phi_log[:n],
                mode="lines",
                name=f"step {step}",
            )
        )

        fig.update_layout(
            title=f"Log spectral density at step {step}",
            template="plotly_white",
            width=850,
            height=550,
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=60),
        )
        fig.update_xaxes(title_text="Eigenvalue [λ]", range=[np.log10(xlim[0]), np.log10(xlim[1])] if xlim[0] > 0 else None)
        fig.update_yaxes(title_text="log Density [log φ(λ)]", range=list(y_lim))

        out_path = _add_plotly_prefix(
            spectra_path / f"log_spectrum_step_{step}.html"
        )

        fig.write_html(
            out_path,
            include_plotlyjs="cdn",
        )