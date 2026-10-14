# eigengrok/config.py

"""
Experiment configuration: one frozen dataclass, loaded from YAML.

Every run is reproducible from a config file plus command-line overrides:

    python -m eigengrok.train --config configs/smoke.yaml --set seed=3

and writes its fully resolved config to config_full.json in its run
directory (see eigengrok/outputs.py).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    # Admin
    run_spectral: bool = True
    task: str = "MODULAR"

    # Data
    data_directory: str = field(default_factory=lambda: os.environ.get("EIGENGROK_DATA_DIR", "./data"))
    train_points: int = 2000
    batch_size: int = 300

    # Modular arithmetic task
    modulus: int = 91
    train_frac: float = 0.3

    # Model
    input_dim: int = 784
    output_dim: int = 10
    depth: int = 3
    width: int = 200
    num_heads: int = 4
    activation: str = "ReLU"
    initialization_scale: float = 1.0

    # Hessian spectral snapshots (eigengrok/hessian/)
    spectral_m: int = 50               # Lanczos steps per probe
    spectral_k: int = 8                # Rademacher probes; the same probes at every snapshot of a run (seeded by `seed`)
    spectral_max_points: int = 8192    # the Hessian is of the loss on the whole training set, or on a seeded subset of this size if larger
    spectral_growth: float = 1.15      # snapshot schedule: geometric from step 1 up to spectral_every ...
    spectral_every: int = 1000         # ... then every spectral_every steps (eigengrok/train/spectral_schedule.py)
    spectral_sigma_frac: float = 0.01  # plotted density only: Gaussian bandwidth as a fraction of the eigenvalue range

    # Optimization
    optimization_steps: int = 200_000
    optimizer: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 0.01
    adam_beta1: float = 0.9            # Adam/AdamW only; both betas default to PyTorch's
    adam_beta2: float = 0.999
    grad_clip: float | None = None     # max global gradient norm per step (clip_grad_norm_); None = no clipping
    loss_function: str = "CrossEntropy"

    # Logging/saving
    log_every: int = 150
    eval_every: int = 1000
    checkpoint_every: int = 1000  # independent of eval_every -- see eigengrok/train/loop.py; 0 = no checkpoints
    output_dir: str = field(default_factory=lambda: os.environ.get("EIGENGROK_OUTPUT_DIR", "./results"))
    seed: int = 0
    dtype: str = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch_dtype(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], *, strict: bool = True) -> ExperimentConfig:
        """
        Builds a config from plain values (YAML, JSON, `--set` strings), coercing
        each to its field's type. With `strict`, an unknown key is an error, so a
        typo in a config file can't silently fall back to a default. Old runs'
        config_full.json files carry since-removed fields (e.g. fgsm_epsilon,
        test_mode), so they are read with `strict=False`, which drops those.
        """
        types = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(values) - set(types))
        if unknown and strict:
            raise ValueError(f"Unknown config key(s) {unknown}. Valid keys: {sorted(types)}")
        coerced = {k: _coerce(k, types[k], v) for k, v in values.items() if k in types}
        cfg = cls(**coerced)
        torch_dtype(cfg.dtype)  # fail early on a bad dtype name
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
        """Loads a YAML config, then applies `key=value` overrides on top."""
        values = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(values, dict):
            raise ValueError(f"{path} must contain a YAML mapping of config keys, got {type(values).__name__}")
        for item in overrides:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"Override {item!r} must look like key=value")
            values[key.strip()] = value.strip()
        return cls.from_dict(values)


def torch_dtype(name: str) -> torch.dtype:
    """'float32' -> torch.float32. Also accepts 'torch.float32', the form older config_full.json files used."""
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype {name!r}")
    return dtype


_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}
_NONE = {"none", "null", ""}


def _coerce(key: str, type_name: str, value: Any) -> Any:
    """Coerces one config value to its dataclass field type (given as a string under postponed annotations)."""
    if value is None:
        return None
    optional = type_name.endswith(" | None")
    if optional:
        if isinstance(value, str) and value.strip().lower() in _NONE:
            return None
        type_name = type_name.removesuffix(" | None")
    try:
        if type_name == "bool":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ValueError(f"not a boolean: {value!r}")
        if type_name == "int":
            if isinstance(value, bool):
                raise ValueError(f"not an integer: {value!r}")
            as_float = float(value)  # accepts "1e5" and 150000.0
            if not as_float.is_integer():
                raise ValueError(f"not an integer: {value!r}")
            return int(as_float)
        if type_name == "float":
            if isinstance(value, bool):
                raise ValueError(f"not a number: {value!r}")
            return float(value)  # YAML 1.1 reads "1e-3" as a string; float() handles it
        if type_name == "str":
            return str(value)
    except ValueError as e:
        raise ValueError(f"Config key {key!r}: {e}") from None
    raise TypeError(f"Config key {key!r} has unsupported field type {type_name!r}")
