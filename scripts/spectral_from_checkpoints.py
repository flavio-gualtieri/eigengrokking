#!/usr/bin/env python3
# scripts/spectral_from_checkpoints.py

"""
Post-hoc Hessian spectral analysis from saved checkpoints.

Loads a run's checkpoints (see project_io/dir_making.py for the directory
layout) at a chosen set of steps and computes the same spectral observables
training/loop.py computes live -- analysis.eigenthings.estimate_density +
analysis.spectral_observables.compute_spectral_observables_with_stderr --
writing one row per step to a parquet file.

Exists for runs trained with run_spectral=False (e.g. the modulus=91
weight-decay sweep driven by scripts/run_grok_mod91.py), where the Hessian
spectrum was never logged during training: this recovers it after the fact
from checkpoints alone, with no retraining.

Usage:
    python scripts/spectral_from_checkpoints.py <run_dir> --steps 3000:7000:100 --out spectra.parquet
    python scripts/spectral_from_checkpoints.py <run_dir> --steps 100,200,500 --out spectra.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from analysis.eigenthings import estimate_density  # noqa: E402
from analysis.spectral_observables import compute_spectral_observables_with_stderr  # noqa: E402
from configs.base import ExperimentConfig  # noqa: E402
from tasks.registry import TASKS  # noqa: E402


def _checkpoint_steps(ckpt_dir: Path) -> List[int]:
    steps = []
    for p in ckpt_dir.glob("checkpoint_step*.pt"):
        m = re.search(r"checkpoint_step(\d+)\.pt$", p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _parse_steps(spec: str, available: List[int]) -> List[int]:
    """
    Parses --steps, in one of two forms:
      "start:stop:stride" -- every step from start to stop inclusive
      "s1,s2,s3"          -- an explicit list

    Either way, the result is filtered down to steps with a checkpoint
    actually on disk -- not every requested step need exist.
    """
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Range steps must be 'start:stop:stride', got {spec!r}")
        start, stop, stride = (int(p) for p in parts)
        wanted = list(range(start, stop + 1, stride))
    else:
        wanted = [int(s) for s in spec.split(",") if s]

    available_set = set(available)
    return [s for s in wanted if s in available_set]


def _load_run_config(run_dir: Path, **overrides: Any) -> ExperimentConfig:
    """
    Rebuilds an ExperimentConfig good enough to reconstruct the model and
    data a run's checkpoints came from. Reads config_full.json (written by
    project_io/dir_making.make_dirs at training time) rather than trusting a
    checkpoint's embedded config -- dropping unknown keys so a field that's
    since been renamed or removed (e.g. the old `spectral_sigma` ->
    `spectral_sigma_frac`) doesn't crash construction of the *current*
    dataclass.
    """
    saved = json.loads((run_dir / "config_full.json").read_text())

    current_fields = {f.name for f in fields(ExperimentConfig)}
    filtered = {k: v for k, v in saved.items() if k in current_fields}
    filtered.update(overrides)

    # This script recomputes the spectrum itself, from checkpoints -- it is
    # never the training loop's live SpectralSchedule-driven logging, so the
    # reconstructed config should never claim run_spectral=True regardless
    # of what the original training run used.
    filtered["run_spectral"] = False

    return ExperimentConfig(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run base dir (contains checkpoints/, config_full.json)")
    parser.add_argument("--steps", required=True, help="'start:stop:stride' or comma-separated step list")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--spectral_m", type=int, default=None, help="Override spectral_m (default: configs/base.py)")
    parser.add_argument("--spectral_k", type=int, default=None, help="Override spectral_k (default: configs/base.py)")
    parser.add_argument("--spectral_sigma_frac", type=float, default=None)
    parser.add_argument("--spectral_batch_size", type=int, default=None)
    parser.add_argument("--probe_seed", type=int, default=None, help="Default: the run's own seed")
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    args = parser.parse_args()

    # This model is tiny (hundreds of thousands of params); PyTorch's default
    # intra-op thread count (one per detected CPU) causes pathological
    # thread-contention overhead on shared/virtualized nodes for workloads
    # this small, dwarfing any parallelism benefit.
    torch.set_num_threads(4)

    run_dir = args.run_dir.resolve()
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints/ under {run_dir}")

    available_steps = _checkpoint_steps(ckpt_dir)
    if not available_steps:
        raise FileNotFoundError(f"No checkpoint_step*.pt files under {ckpt_dir}")

    steps = _parse_steps(args.steps, available_steps)
    if not steps:
        raise ValueError(
            f"None of the requested steps ({args.steps!r}) have a checkpoint on disk. "
            f"Available range: [{available_steps[0]}, {available_steps[-1]}]"
        )

    overrides: Dict[str, Any] = {}
    if args.spectral_m is not None:
        overrides["spectral_m"] = args.spectral_m
    if args.spectral_k is not None:
        overrides["spectral_k"] = args.spectral_k
    if args.spectral_sigma_frac is not None:
        overrides["spectral_sigma_frac"] = args.spectral_sigma_frac
    if args.spectral_batch_size is not None:
        overrides["spectral_batch_size"] = args.spectral_batch_size

    cfg = _load_run_config(run_dir, **overrides)
    probe_seed = args.probe_seed if args.probe_seed is not None else cfg.seed

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Run: {run_dir}")
    print(
        f"Config: task={cfg.task} modulus={getattr(cfg, 'modulus', None)} depth={cfg.depth} width={cfg.width} "
        f"wd={cfg.weight_decay} init={cfg.initialization_scale} seed={cfg.seed}"
    )
    print(
        f"Spectral params: m={cfg.spectral_m} k={cfg.spectral_k} sigma_frac={cfg.spectral_sigma_frac} "
        f"batch_size={cfg.spectral_batch_size} probe_seed={probe_seed}"
    )
    print(f"Steps ({len(steps)}): {steps[0]}..{steps[-1]}")
    print(f"Device: {device}")

    task = TASKS[cfg.task](cfg)
    model = task.build_model(device)

    data = task.build_data()
    train_ds = data["train_ds"]
    spectral_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=min(cfg.spectral_batch_size, len(train_ds)),
        shuffle=False,
    )
    x_spectral, y_spectral = next(iter(spectral_loader))
    x_spectral = x_spectral.to(device).float()
    y_spectral = y_spectral.to(device)

    rows: List[Dict[str, Any]] = []

    for step in steps:
        ckpt = torch.load(ckpt_dir / f"checkpoint_step{step}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        logits_spec = model(x_spectral)
        loss_spec = task.compute_training_loss(logits_spec, y_spectral, device)

        spec = estimate_density(
            model=model,
            m=cfg.spectral_m,
            k=cfg.spectral_k,
            sigma_frac=cfg.spectral_sigma_frac,
            loss=loss_spec,
            probe_seed=probe_seed,
        )

        obs, obs_stderr = compute_spectral_observables_with_stderr(
            probe_nodes=spec.probe_nodes,
            probe_weights=spec.probe_weights,
            n_params=spec.n_params,
        )

        row: Dict[str, Any] = {"step": step}
        for f in fields(obs):
            row[f.name] = getattr(obs, f.name)
            row[f"{f.name}_stderr"] = getattr(obs_stderr, f.name)
        rows.append(row)

        print(
            f"step {step:7d}: top_eig={obs.top_eig:.3e}±{obs_stderr.top_eig:.1e}, "
            f"bulk_edge={obs.bulk_edge:.3e}, trace={obs.trace:.3e}, "
            f"eff_rank={obs.effective_rank:.2f}, neg_mass={obs.negative_mass:.3e}, "
            f"cond={obs.conditioning:.3e}"
        )

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
