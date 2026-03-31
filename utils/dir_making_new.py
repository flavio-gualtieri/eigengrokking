from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any
import json
import re


def _fmt_float(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.0e}" if (abs(x) < 1e-3 or abs(x) >= 1e3) else f"{x:g}"


def _slugify(s: str) -> str:
    s = s.strip().replace(" ", "").replace("/", "-")
    return re.sub(r"[^a-zA-Z0-9_.=\-]+", "", s)


def _identity_dict(cfg) -> Dict[str, Any]:
    """Only include parameters that define the sweep."""
    return {
        "dataset": str(cfg.dataset),
        "depth": int(cfg.depth),
        "width": int(cfg.width),
        "initialization_scale": float(cfg.initialization_scale),
    }


def make_dirs(cfg, test_mode: bool = False, toy_mode: bool = False) -> dict[str, Path]:
    out_root = Path(cfg.output_dir)

    if test_mode:
        out_root = out_root / "TEST"
    if toy_mode:
        out_root = out_root / "TOY"

    id_dict = _identity_dict(cfg)

    # Only encode varied parameters in the path
    group_parts = [
        str(cfg.dataset),
        f"depth={cfg.depth}",
        f"width={cfg.width}",
        f"init={_fmt_float(cfg.initialization_scale)}",
    ]

    base = out_root.joinpath(*map(_slugify, group_parts))

    layer_dir = base / "layer_outputs"
    ckpt_dir = base / "checkpoints"
    fig_dir = base / "figures"

    for d in (layer_dir, ckpt_dir, fig_dir):
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
    }