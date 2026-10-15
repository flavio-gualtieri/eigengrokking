#!/usr/bin/env python3
# scripts/spectral_from_checkpoints.py

"""
Post-hoc Hessian spectral analysis from saved checkpoints.

Loads a run's checkpoints (see eigengrok/outputs.py for the directory
layout) at a chosen set of steps and computes the same spectral statistics
eigengrok/train/loop.py computes live (eigengrok/hessian/spectrum.py, on the
same fixed evaluation batch: the whole training set unless it exceeds
spectral_max_points), writing one row per step to a parquet file: each
statistic plus its `<name>_stderr` over probes. Every checkpoint uses the same
probes (probe seed = the run's seed), so changes between checkpoints are far
more precise than the values themselves (reports/spectral_validation.md).

--operators picks the curvature operators: `hessian` (columns named as the
statistics, e.g. `top_eig`) and/or `gauss_newton` (prefixed `gn_`, e.g.
`gn_top_eig`). Each row also records the settings (m, k, probe_seed,
n_points) and `seconds`, the wall time for that checkpoint.

Exists for runs trained with run_spectral=False (e.g. the modulus=91
weight-decay sweep, configs/mod91_wd_sweep.yaml), where the Hessian
spectrum was never logged during training: this recovers it after the fact
from checkpoints alone, with no retraining.

Usage:
    python scripts/spectral_from_checkpoints.py <run_dir> --steps 3000:7000:100 --out spectra.parquet
    python scripts/spectral_from_checkpoints.py <run_dir> --steps 100,200,500 --out spectra.parquet
    python scripts/spectral_from_checkpoints.py <run_dir> --steps all --operators hessian,gauss_newton --out spectra.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from eigengrok.hessian.operators import CurvatureOperators
from eigengrok.hessian.snapshot import OPERATORS, fixed_eval_batch, operator_stats
from eigengrok.config import ExperimentConfig
from eigengrok.tasks.registry import TASKS


def _checkpoint_steps(ckpt_dir: Path) -> List[int]:
    steps = []
    for p in ckpt_dir.glob("checkpoint_step*.pt"):
        m = re.search(r"checkpoint_step(\d+)\.pt$", p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _parse_steps(spec: str, available: List[int]) -> List[int]:
    """
    Parses --steps, in one of three forms:
      "start:stop:stride" -- every step from start to stop inclusive
      "s1,s2,s3"          -- an explicit list
      "all"               -- every checkpoint on disk

    Either way, the result is filtered down to steps with a checkpoint
    actually on disk -- not every requested step need exist.
    """
    if spec == "all":
        return list(available)
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
    eigengrok/outputs.py's make_dirs at training time) rather than trusting a
    checkpoint's embedded config -- dropping unknown keys (strict=False) so
    a field that's since been renamed or removed (e.g. the old
    `spectral_sigma` -> `spectral_sigma_frac`) doesn't crash construction of
    the *current* dataclass.
    """
    saved = json.loads((run_dir / "config_full.json").read_text())
    saved.update(overrides)

    # This script recomputes the spectrum itself, from checkpoints -- it is
    # never the training loop's live SpectralSchedule-driven logging, so the
    # reconstructed config should never claim run_spectral=True regardless
    # of what the original training run used.
    saved["run_spectral"] = False

    return ExperimentConfig.from_dict(saved, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Run base dir (contains checkpoints/, config_full.json)")
    parser.add_argument("--steps", required=True, help="'start:stop:stride', a comma-separated step list, or 'all'")
    parser.add_argument("--operators", default="hessian",
                        help=f"Comma-separated, from {sorted(OPERATORS)} (default: hessian)")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--spectral_m", type=int, default=None, help="Override spectral_m (default: eigengrok/config.py)")
    parser.add_argument("--spectral_k", type=int, default=None, help="Override spectral_k (default: eigengrok/config.py)")
    parser.add_argument("--spectral_sigma_frac", type=float, default=None)
    parser.add_argument("--spectral_max_points", type=int, default=None)
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
    if args.spectral_max_points is not None:
        overrides["spectral_max_points"] = args.spectral_max_points

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
        f"max_points={cfg.spectral_max_points} probe_seed={probe_seed}"
    )
    print(f"Steps ({len(steps)}): {steps[0]}..{steps[-1]}")
    print(f"Device: {device}")

    task = TASKS[cfg.task](cfg)
    model = task.build_model(device)

    data = task.build_data()
    x_hessian, y_hessian = fixed_eval_batch(data["train_ds"], cfg.spectral_max_points, seed=cfg.seed, device=device)
    hessian_loss = lambda logits, labels: task.compute_training_loss(logits, labels, device)  # noqa: E731

    operators = [o.strip() for o in args.operators.split(",") if o.strip()]
    unknown = sorted(set(operators) - set(OPERATORS))
    if unknown:
        raise ValueError(f"Unknown operator(s) {unknown}; options: {sorted(OPERATORS)}")
    prefix = {"hessian": "", "gauss_newton": "gn_", "functional": "fn_"}

    rows: List[Dict[str, Any]] = []

    for step in steps:
        ckpt = torch.load(ckpt_dir / f"checkpoint_step{step}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        ops = CurvatureOperators(model, x_hessian, y_hessian, hessian_loss)
        row: Dict[str, Any] = {"step": step, "m": cfg.spectral_m, "k": cfg.spectral_k, "probe_seed": probe_seed,
                               "n_points": len(y_hessian)}
        for op in operators:
            stats, stderr = operator_stats(ops, op, m=cfg.spectral_m, k=cfg.spectral_k, probe_seed=probe_seed)
            for name, value in stats.items():
                row[f"{prefix[op]}{name}"] = value
                row[f"{prefix[op]}{name}_stderr"] = stderr[name]
        if device.type == "cuda":
            torch.cuda.synchronize()
        row["seconds"] = time.perf_counter() - start
        rows.append(row)

        print(
            f"step {step:7d}: {row['seconds']:.2f}s  "
            + "  ".join(f"{prefix[op] or 'H_'}top_eig={row[prefix[op] + 'top_eig']:.3e} "
                        f"{prefix[op] or 'H_'}trace={row[prefix[op] + 'trace']:.3e} "
                        f"{prefix[op] or 'H_'}eff_rank={row[prefix[op] + 'effective_rank']:.1f}" for op in operators)
        )

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
