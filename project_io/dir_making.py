from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any
import json
import re
import hashlib


def _fmt_float(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.0e}" if (abs(x) < 1e-3 or abs(x) >= 1e3) else f"{x:g}"


def _slugify(s: str) -> str:
    s = s.strip().replace(" ", "").replace("/", "-")
    return re.sub(r"[^a-zA-Z0-9_.=\-]+", "", s)


def _identity_dict(cfg) -> Dict[str, Any]:
    # Minimal "same experiment" definition per your request
    return {
        "dataset": str(cfg.dataset),
        "initialization_scale": float(cfg.initialization_scale),
        "weight_decay": float(cfg.weight_decay),
        "epochs": int(cfg.optimization_steps),
        "spectrum": bool(getattr(cfg, "run_spectral", False)),
        # keep seed so different seeds don't overwrite each other
        "seed": int(cfg.seed),
    }


def make_dirs(cfg, test_mode: bool = False):
    out_root = Path(cfg.output_dir)

    # ---- TEST MODE ----
    if test_mode:
        out_root = out_root / "TEST"

    id_dict = _identity_dict(cfg)

    # ----- PATH STRUCTURE -----
    group_parts = [str(cfg.dataset)]

    # only include marker when spectral is OFF
    if not getattr(cfg, "run_spectral", False):
        group_parts.append("no_spectrum")

    group_parts += [
        f"init={_fmt_float(cfg.initialization_scale)}",
        f"wd={_fmt_float(cfg.weight_decay)}",
    ]

    # hash removed → same config overwrites
    leaf = f"seed={cfg.seed}"

    base = out_root.joinpath(*map(_slugify, group_parts), _slugify(leaf))

    layer_dir = base / "layer_outputs"
    ckpt_dir = base / "checkpoints"
    fig_dir = base / "figures"
    spectra_dir = base / "spectral_snapshots"

    dirs_to_make = [layer_dir, ckpt_dir, fig_dir]
    if getattr(cfg, "run_spectral", False):
        dirs_to_make.append(spectra_dir)

    for d in dirs_to_make:
        d.mkdir(parents=True, exist_ok=True)

    (base / "identity.json").write_text(
        json.dumps(id_dict, indent=2),
        encoding="utf-8",
    )

    (base / "config_full.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "base": base,
        "layer": layer_dir,
        "ckpt": ckpt_dir,
        "fig": fig_dir,
        "spectra": spectra_dir,
    }