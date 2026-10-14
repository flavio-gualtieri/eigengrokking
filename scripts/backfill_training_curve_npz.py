#!/usr/bin/env python3
# scripts/backfill_training_curve_npz.py

"""
Backfills figures/training_data_*.npz for wd=0.6 seeds 1-9.

That sweep predates eigengrok/outputs.py's current logging path -- only
seed=0 has a training_data_*.npz (like config_full.json, hand-patched just
for the existing gate figure). eigengrok/analysis/transitions.py::load_and_join
requires that npz (train_steps, norms, last_layer_norms, test_accuracies)
for every run in the effective_rank multi-seed replication
(eigengrok/analysis/aggregate_effective_rank.py), so seeds 1-9 need it backfilled
before that script can run.

norms/last_layer_norms are recomputed exactly (not approximated) by loading
each checkpoint's state dict into the same architecture and calling
eigengrok/train/loop.py::_weight_norms -- deterministic given the checkpoint, no
retraining or forward pass needed. test_accuracies (and the extra
train_accuracies/losses columns, for parity with the schema
scripts/build_wd_sweep_accuracy_csv.py and analysis/aggregate_seeds.py
expect) come from the already-committed reports/wd_sweep_accuracy_curves.csv.

CPU-only, no GPU needed (state-dict loads and norm computation, no forward
pass) -- run directly, not via SLURM.

Usage: python scripts/backfill_training_curve_npz.py [--wd 0.6] [--seeds 1-9]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from eigengrok.config import ExperimentConfig
from eigengrok.tasks.registry import TASKS
from eigengrok.train.loop import _weight_norms

ACCURACY_CSV = Path("reports/wd_sweep_accuracy_curves.csv")


def _checkpoint_steps(ckpt_dir: Path) -> List[int]:
    steps = []
    for p in ckpt_dir.glob("checkpoint_step*.pt"):
        m = re.search(r"checkpoint_step(\d+)\.pt$", p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _load_run_config(run_dir: Path) -> ExperimentConfig:
    saved = json.loads((run_dir / "config_full.json").read_text())
    saved["run_spectral"] = False
    return ExperimentConfig.from_dict(saved, strict=False)


def backfill_one(wd: str, seed: int, *, overwrite: bool = False) -> None:
    run_dir = Path(f"results/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd={wd}/seed={seed}")
    fig_dir = run_dir / "figures"
    out_path = fig_dir / f"training_data_dataset=MODULAR_p-91_alpha=1_wd={wd}.npz"
    if out_path.exists() and not overwrite:
        print(f"wd={wd} seed={seed}: {out_path} already present, skipping")
        return

    ckpt_dir = run_dir / "checkpoints"
    steps = _checkpoint_steps(ckpt_dir)
    if not steps:
        print(f"wd={wd} seed={seed}: no checkpoints under {ckpt_dir}, skipping")
        return

    acc = pd.read_csv(ACCURACY_CSV)
    mask = (
        (acc["modulus"] == 91) & (acc["depth"] == 1) & (acc["width"] == 128) & (acc["init"] == 1)
        & np.isclose(acc["wd"], float(wd), atol=1e-9) & (acc["seed"] == seed)
    )
    run_acc = acc.loc[mask].set_index("step")
    if run_acc.empty:
        raise ValueError(f"No rows in {ACCURACY_CSV} for wd={wd} seed={seed}")

    # Only steps with both a checkpoint and an accuracy-CSV row -- same
    # inner-join discipline as eigengrok/analysis/transitions.py::load_and_join.
    steps = [s for s in steps if s in run_acc.index]
    if not steps:
        raise ValueError(f"No checkpoint step for wd={wd} seed={seed} matches {ACCURACY_CSV}'s step grid")

    cfg = _load_run_config(run_dir)
    device = torch.device("cpu")
    task = TASKS[cfg.task](cfg)
    model = task.build_model(device)

    wnorms, ll_norms = [], []
    for step in steps:
        ckpt = torch.load(ckpt_dir / f"checkpoint_step{step}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        wn, lwn = _weight_norms(model)
        wnorms.append(wn)
        ll_norms.append(lwn)

    train_steps = np.array(steps, dtype=np.int64)
    test_acc = run_acc.loc[steps, "test_accuracy"].to_numpy()
    train_acc = run_acc.loc[steps, "train_accuracy"].to_numpy()
    test_loss = run_acc.loc[steps, "test_loss"].to_numpy() if "test_loss" in run_acc.columns else None
    train_loss = run_acc.loc[steps, "train_loss"].to_numpy() if "train_loss" in run_acc.columns else None

    fig_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs: Dict[str, Any] = dict(
        train_steps=train_steps,
        test_steps=train_steps,
        norms=np.array(wnorms, dtype=np.float64),
        last_layer_norms=np.array(ll_norms, dtype=np.float64),
        test_accuracies=test_acc,
        train_accuracies=train_acc,
    )
    if test_loss is not None:
        save_kwargs["test_losses"] = test_loss
    if train_loss is not None:
        save_kwargs["train_losses"] = train_loss

    np.savez(out_path, **save_kwargs)
    print(f"wd={wd} seed={seed}: wrote {out_path} ({len(steps)} steps)")


def _parse_seeds(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wd", default="0.6")
    parser.add_argument("--seeds", default="1-9", help="'a-b' range or comma list")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for seed in _parse_seeds(args.seeds):
        backfill_one(args.wd, seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
