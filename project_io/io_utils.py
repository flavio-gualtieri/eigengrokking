# io_utils.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt


def _fmt_float(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.0e}" if (abs(x) < 1e-3 or abs(x) >= 1e3) else f"{x:g}"


def _dataset_label(cfg) -> str:
    if cfg.dataset == "MODULAR":
        return f"MODULAR(p={cfg.modulus})"
    return str(cfg.dataset)


def training_plot_stem(cfg) -> str:
    dataset = _dataset_label(cfg).replace("(", "_").replace(")", "").replace("=", "-")
    alpha = _fmt_float(float(cfg.initialization_scale))
    weight_decay = _fmt_float(float(cfg.weight_decay))
    return f"dataset={dataset}_alpha={alpha}_wd={weight_decay}"


def _to_numpy(arr):
    if arr is None:
        return None

    # Single tensor
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()

    # List/tuple of tensors (common for history lists)
    if isinstance(arr, (list, tuple)) and len(arr) > 0 and isinstance(arr[0], torch.Tensor):
        # If they are scalar tensors, convert to float array
        if all(isinstance(x, torch.Tensor) and x.numel() == 1 for x in arr):
            return np.asarray([float(x.detach().cpu().item()) for x in arr], dtype=float)
        # Otherwise convert each to numpy and stack if possible
        return np.asarray([x.detach().cpu().numpy() for x in arr], dtype=object)

    return np.asarray(arr)


def save_spectral_snapshots(
        *,
        spectra_dir: Path,
        phi_store: Dict[int, torch.Tensor],
        t: Optional[torch.Tensor] = None,
        t_store: Optional[Dict[int, torch.Tensor]] = None,
        dpi: int = 200,
    ) -> None:
    if t_store is not None and len(t_store) > 0:
        t_mins = []
        t_maxs = []
        for t_step in t_store.values():
            t_step_cpu = t_step.detach().cpu().numpy()
            if t_step_cpu.size > 0:
                t_mins.append(float(t_step_cpu[0]))
                t_maxs.append(float(t_step_cpu[-1]))
        xlim = (min(t_mins), max(t_maxs)) if t_mins and t_maxs else None
    else:
        if t is None:
            raise ValueError("Either t or t_store must be provided for spectral snapshots.")
        t_cpu = t.detach().cpu().numpy()
        xlim = (float(t_cpu[0]), float(t_cpu[-1]))

    # Fix y-limits once, based on FIRST snapshot
    first_step = sorted(phi_store)[0]
    phi0 = phi_store[first_step].detach().cpu().numpy()
    y_min = float(np.nanmin(phi0))
    y_max = float(np.nanmax(phi0))

    # Small padding so it doesn't sit on the border
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_lim = (y_min - pad, y_max + pad)

    for step in sorted(phi_store):
        phi = phi_store[step].detach().cpu().numpy()
        if t_store is not None and step in t_store:
            t_cpu = t_store[step].detach().cpu().numpy()
        else:
            if t is None:
                raise ValueError("Missing t for spectral snapshot plotting.")
            t_cpu = t.detach().cpu().numpy()

        if t_cpu.shape[0] != phi.shape[0]:
            n = min(t_cpu.shape[0], phi.shape[0])
            t_cpu = t_cpu[:n]
            phi = phi[:n]

        plt.figure()
        plt.plot(t_cpu, phi)
        plt.xlabel("Eigenvalue [λ]")
        plt.ylabel("Density [φ(λ)]")
        plt.title(f"Spectral density at step {step}")
        if xlim is not None:
            plt.xlim(*xlim)
        plt.ylim(*y_lim)
        plt.tight_layout()
        plt.savefig(spectra_dir / f"spectrum_step_{step}.png", dpi=dpi)
        plt.close()


def save_layer_outputs_npz(
        save_path: Path,
        layer_outputs,
        labels=None,
        intrinsic_dims: Optional[np.ndarray] = None,
        l2n2_intrinsic_dims: Optional[np.ndarray] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
    """
    Save per-layer outputs and metadata into a compressed .npz file.

    Parameters
    ----------
    save_path : Path
    layer_outputs : list[tensor or array]
        Outputs for each layer.
    labels : tensor or array
    intrinsic_dims : array or None
    l2n2_intrinsic_dims : array or None
    extra_meta : optional dict (saved as npz entries if JSON-safe)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: Dict[str, Any] = {}

    for i, arr in enumerate(layer_outputs):
        save_dict[f"layer_{i}"] = _to_numpy(arr)

    if labels is not None:
        save_dict["labels"] = _to_numpy(labels)

    if intrinsic_dims is not None:
        save_dict["intrinsic_dims"] = np.asarray(intrinsic_dims)

    if l2n2_intrinsic_dims is not None:
        save_dict["l2n2_intrinsic_dims"] = np.asarray(l2n2_intrinsic_dims)

    if extra_meta:
        # store simple scalars/arrays; complex objects are stringified
        for k, v in extra_meta.items():
            if isinstance(v, (int, float, str, bool, np.ndarray)):
                save_dict[f"meta_{k}"] = v
            else:
                save_dict[f"meta_{k}"] = str(v)

    np.savez_compressed(save_path, **save_dict)


def save_checkpoint(
        path: Path,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict[str, Any],
        config: Optional[Any] = None,
    ) -> None:
    """
    Save model + optimizer + training history checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if config is not None:
        if is_dataclass(config):
            config_obj = asdict(config)
        else:
            config_obj = config
    else:
        config_obj = None

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config_obj,
        },
        path,
    )


def save_training_data_npz(
        path: Path,
        history: Dict[str, Any],
    ) -> None:
    """
    Save training logs (steps, accuracies, norms, etc.) to .npz
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    for k, v in history.items():
        try:
            save_dict[k] = np.asarray(v)
        except Exception:
            save_dict[k] = np.asarray([str(v)])

    np.savez_compressed(path, **save_dict)


def _plot_training_with_right_axis(
        path: Path,
        cfg,
        history: Dict[str, Any],
        right_steps: Sequence,
        right_values: Sequence,
        right_ylabel: str,
        right_legend_label: str,
    ) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch data from history (left axis lines are ALWAYS the same)
    train_log_steps = history.get("train_log_steps", [])
    test_log_steps = history.get("test_log_steps", [])
    train_accuracies = history.get("train_accuracies", [])
    test_accuracies = history.get("test_accuracies", [])
    adv_test_accuracies = history.get("adv_test_accuracies", [])

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(train_log_steps) > 0:
        ax.plot(_to_numpy(train_log_steps), _to_numpy(train_accuracies), label="train", color="blue")
    if len(test_log_steps) > 0:
        ax.plot(_to_numpy(test_log_steps), _to_numpy(test_accuracies), label="test", color="red")
        if len(adv_test_accuracies) == len(test_log_steps):
            ax.plot(
                _to_numpy(test_log_steps),
                _to_numpy(adv_test_accuracies),
                label=f"adv test (FGSM ε={cfg.fgsm_epsilon})",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Accuracy")

    # Right axis (ONLY thing that changes across the extra plots)
    ax2 = ax.twinx()
    right_steps_np = _to_numpy(right_steps)
    right_values_np = _to_numpy(right_values)

    if right_steps_np is not None and right_values_np is not None:
        if len(right_steps_np) > 0 and len(right_values_np) == len(right_steps_np):
            ax2.plot(
                right_steps_np,
                right_values_np,
                color="purple",
                label=right_legend_label,
                linestyle="-",
            )
            ax2.set_ylabel(right_ylabel)

    # Combined legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if (handles1 + handles2):
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    dataset_info = _dataset_label(cfg)
    train_info = (
        f"{cfg.train_points:.2f} frac"
        if isinstance(cfg.train_points, float)
        else f"{cfg.train_points} pts"
    )
    focus_info = (
        f"dataset={dataset_info}, "
        f"alpha={_fmt_float(float(cfg.initialization_scale))}, "
        f"WD={_fmt_float(float(cfg.weight_decay))}"
    )

    ax.set_title(
        f"{focus_info}\n"
        f"depth={cfg.depth}, width={cfg.width}, act={cfg.activation}, "
        f"train={train_info}, bs={cfg.batch_size}, steps={cfg.optimization_steps}",
        fontsize=10,
    )

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_plot_png(
        path: Path,
        cfg,
        history: Dict[str, Any],
    ) -> None:
    """
    Save the original accuracy plot PLUS three extra variants where only the
    right-axis series changes (kappa, phi_mean, phi_var).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    eig_log_steps = history.get("eig_log_steps", [])
    eig_mass_gt1 = history.get("eig_mass_gt1", [])
    kappa = history.get("kappa", [])
    phi_mean = history.get("phi_mean", [])
    phi_var = history.get("phi_var", [])

    stem = path.stem
    suffix = path.suffix
    out_dir = path.parent

    # 1) Original (eig_mass_gt1)
    _plot_training_with_right_axis(
        out_dir / f"{stem}{suffix}",
        cfg,
        history,
        right_steps=eig_log_steps,
        right_values=eig_mass_gt1,
        right_ylabel="Mass fraction with λ > 1",
        right_legend_label="eig mass (λ>1)",
    )

    # 2) Extra: kappa
    _plot_training_with_right_axis(
        out_dir / f"{stem}_kappa{suffix}",
        cfg,
        history,
        right_steps=eig_log_steps,
        right_values=kappa,
        right_ylabel="kappa",
        right_legend_label="kappa",
    )

    # 3) Extra: phi_mean
    _plot_training_with_right_axis(
        out_dir / f"{stem}_phi_mean{suffix}",
        cfg,
        history,
        right_steps=eig_log_steps,
        right_values=phi_mean,
        right_ylabel="phi_mean",
        right_legend_label="phi_mean",
    )

    # 4) Extra: phi_var
    _plot_training_with_right_axis(
        out_dir / f"{stem}_phi_var{suffix}",
        cfg,
        history,
        right_steps=eig_log_steps,
        right_values=phi_var,
        right_ylabel="phi_var",
        right_legend_label="phi_var",
    )


def save_log_spectral_snapshots(
        *,
        spectra_dir: Path,
        phi_store: Dict[int, torch.Tensor],
        t: Optional[torch.Tensor] = None,
        t_store: Optional[Dict[int, torch.Tensor]] = None,
        dpi: int = 200,
        eps: float = 1e-12,
    ) -> None:
    """
    Saves per-step snapshots of log spectral density:
        x-axis: eigenvalue λ (from t)
        y-axis: log φ(λ)

    Uses fixed x-limits from t, and fixed y-limits computed from the FIRST snapshot
    after applying a safe log transform (clip to eps).
    """
    spectra_dir.mkdir(parents=True, exist_ok=True)

    if t_store is not None and len(t_store) > 0:
        t_mins = []
        t_maxs = []
        for t_step in t_store.values():
            t_step_cpu = t_step.detach().cpu().numpy()
            if t_step_cpu.size > 0:
                t_mins.append(float(t_step_cpu[0]))
                t_maxs.append(float(t_step_cpu[-1]))
        xlim = (min(t_mins), max(t_maxs)) if t_mins and t_maxs else None
    else:
        if t is None:
            raise ValueError("Either t or t_store must be provided for log spectral snapshots.")
        t_cpu = t.detach().cpu().numpy()
        xlim = (float(t_cpu[0]), float(t_cpu[-1]))

    # Compute y-limits once from the FIRST snapshot, in LOG space
    first_step = sorted(phi_store)[0]
    phi0 = phi_store[first_step].detach().cpu().numpy()
    phi0_log = np.log(np.clip(phi0, eps, None))

    y_min = float(np.nanmin(phi0_log))
    y_max = float(np.nanmax(phi0_log))

    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_lim = (y_min - pad, y_max + pad)

    for step in sorted(phi_store):
        phi = phi_store[step].detach().cpu().numpy()
        phi_log = np.log(np.clip(phi, eps, None))
        if t_store is not None and step in t_store:
            t_cpu = t_store[step].detach().cpu().numpy()
        else:
            if t is None:
                raise ValueError("Missing t for log spectral snapshot plotting.")
            t_cpu = t.detach().cpu().numpy()

        if t_cpu.shape[0] != phi_log.shape[0]:
            n = min(t_cpu.shape[0], phi_log.shape[0])
            t_cpu = t_cpu[:n]
            phi_log = phi_log[:n]

        plt.figure()
        plt.plot(t_cpu, phi_log)
        plt.xlabel("Eigenvalue [λ]")
        plt.ylabel("log Density [log φ(λ)]")
        plt.title(f"Log spectral density at step {step}")
        if xlim is not None:
            plt.xlim(*xlim)
        plt.ylim(*y_lim)
        plt.tight_layout()
        plt.savefig(spectra_dir / f"log_spectrum_step_{step}.png", dpi=dpi)
        plt.close()
