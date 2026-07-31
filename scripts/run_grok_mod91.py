# scripts/run_grok_mod91.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.base import ExperimentConfig  # noqa: E402
from tasks.mod_add import ModularAddition_task  # noqa: E402
from project_io.dir_making import make_dirs  # noqa: E402


def build_config(
    weight_decay: float,
    optimization_steps: int,
    seed: int,
    initialization_scale: float = 1.0,
) -> ExperimentConfig:
    return ExperimentConfig(
        task="MODULAR",
        modulus=91,
        train_frac=0.3,
        output_dim=91,
        depth=1,
        width=128,
        num_heads=4,
        activation="GELU",
        optimizer="AdamW",
        lr=1e-3,
        weight_decay=weight_decay,
        loss_function="CrossEntropy",
        batch_size=8192,  # > 91**2 * 0.3 (~2484): full-batch, matches configs/experiments.py's _FULL_BATCH pattern
        optimization_steps=optimization_steps,
        initialization_scale=initialization_scale,
        log_every=100,
        eval_every=100,
        checkpoint_every=100,  # fine enough that whatever window the elbow lands in has good checkpoint coverage
        run_spectral=False,    # deferred -- see module docstring
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--optimization_steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initialization_scale", type=float, default=1.0)
    args = parser.parse_args()

    cfg = build_config(args.weight_decay, args.optimization_steps, args.seed, args.initialization_scale)
    dirs = make_dirs(cfg, test_mode=cfg.test_mode)
    print(f"Run directory: {dirs['base']}")

    task = ModularAddition_task(cfg)
    history = task.run()

    test_steps = history["test_steps"]
    test_accs = history["test_accuracies"]
    print("\nDone. Test accuracy at each eval step:")
    for s, a in zip(test_steps, test_accs):
        print(f"  step {s:6d}: {a * 100:5.1f}%")
    print(f"\nAccuracy plot: {dirs['fig']}/training_curve_*_weight_norms.png (and _PLOTLY.html)")
    print(f"Checkpoints:   {dirs['ckpt']}/checkpoint_step*.pt")


if __name__ == "__main__":
    main()
