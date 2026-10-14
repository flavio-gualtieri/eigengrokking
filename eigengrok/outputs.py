from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import json
import re

import numpy as np
import torch


def _fmt_float(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.0e}" if (abs(x) < 1e-3 or abs(x) >= 1e3) else f"{x:g}"


def _slugify(s: str) -> str:
    s = s.strip().replace(" ", "").replace("/", "-")
    return re.sub(r"[^a-zA-Z0-9_.=\-]+", "", s)


def _identity_dict(cfg) -> Dict[str, Any]:
    # Minimal "same experiment" definition (excludes lr, optimizer, train_frac, ...)
    d = {
        "task": str(cfg.task),
        "depth": int(cfg.depth),
        "width": int(cfg.width),
        "initialization_scale": float(cfg.initialization_scale),
        "weight_decay": float(cfg.weight_decay),
        "epochs": int(cfg.optimization_steps),
        "spectrum": bool(getattr(cfg, "run_spectral", False)),
        # keep seed so different seeds don't overwrite each other
        "seed": int(cfg.seed),
    }
    # modulus only means something for MODULAR; without it, e.g. modulus=91
    # and modulus=23 runs at the same depth/width/wd/seed would collide.
    if cfg.task == "MODULAR":
        d["modulus"] = int(cfg.modulus)
    return d


def make_dirs(cfg) -> dict[str, Path]:
    out_root = Path(cfg.output_dir)

    id_dict = _identity_dict(cfg)

    # ----- PATH STRUCTURE -----
    group_parts = [str(cfg.task)]

    # only include marker when spectral is OFF
    if not getattr(cfg, "run_spectral", False):
        group_parts.append("no_spectrum")

    if cfg.task == "MODULAR":
        group_parts.append(f"modulus={cfg.modulus}")

    group_parts += [
        f"depth={cfg.depth}",
        f"width={cfg.width}",
        f"init={_fmt_float(cfg.initialization_scale)}",
        f"wd={_fmt_float(cfg.weight_decay)}",
    ]

    # hash removed → same config overwrites
    leaf = f"seed={cfg.seed}"

    base = out_root.joinpath(*map(_slugify, group_parts), _slugify(leaf))

    ckpt_dir = base / "checkpoints"
    fig_dir = base / "figures"
    spectra_dir = base / "spectral_snapshots"

    dirs_to_make = [ckpt_dir, fig_dir]
    if getattr(cfg, "run_spectral", False):
        dirs_to_make.append(spectra_dir)

    for d in dirs_to_make:
        d.mkdir(parents=True, exist_ok=True)

    (base / "identity.json").write_text(
        json.dumps(id_dict, indent=2),
        encoding="utf-8",
    )

    (base / "config_full.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    return {
        "base": base,
        "ckpt": ckpt_dir,
        "fig": fig_dir,
        "spectra": spectra_dir,
    }


def _dataset_label(cfg) -> str:
    if cfg.task == "MODULAR":
        return f"MODULAR(p={cfg.modulus})"
    return str(cfg.task)


def training_plot_stem(cfg) -> str:
    dataset = _dataset_label(cfg).replace("(", "_").replace(")", "").replace("=", "-")
    alpha = _fmt_float(float(cfg.initialization_scale))
    weight_decay = _fmt_float(float(cfg.weight_decay))
    return f"dataset={dataset}_alpha={alpha}_wd={weight_decay}"


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
