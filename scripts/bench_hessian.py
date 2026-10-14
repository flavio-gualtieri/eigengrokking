#!/usr/bin/env python3
# scripts/bench_hessian.py

"""
Cost of a Hessian snapshot relative to a training step.

For each task setting, times (CUDA-synchronised, after warm-up, median of
repeats):
  - one training step (forward, backward, optimizer step) at the config's batch size
  - one batched HVP on the fixed Hessian batch: reverse-over-reverse on a
    cached first-order graph (eigengrok/hessian/operators.py) and, for
    comparison, torch.func forward-over-reverse
  - one full snapshot at the config's spectral_m / spectral_k
and projects total snapshot cost over a run under the default snapshot
schedule (eigengrok/train/spectral_schedule.py).

Timing does not depend on data values, so inputs are synthetic (no dataset
download needed); shapes match the real tasks.

    python scripts/bench_hessian.py --out reports/hessian_benchmark.md
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path

import torch

from eigengrok.config import ExperimentConfig
from eigengrok.hessian.operators import CurvatureOperators
from eigengrok.hessian.snapshot import hessian_snapshot
from eigengrok.hessian.spectrum import rademacher_probes
from eigengrok.registry import OPTIMIZERS
from eigengrok.tasks.registry import get_task
from eigengrok.train.spectral_schedule import spectral_steps

RUN_LENGTHS = (30_000, 150_000)


def _settings() -> dict[str, tuple[ExperimentConfig, int]]:
    """name -> (config, number of examples in the Hessian batch)."""
    mod = ExperimentConfig.from_yaml("configs/mod91_wd_sweep.yaml", ["run_spectral=true"])
    n_mod = round(mod.train_frac * mod.modulus ** 2)
    mnist = ExperimentConfig(task="MNIST", depth=3, width=200, train_points=2000, batch_size=200,
                             initialization_scale=8.0, loss_function="MSE", weight_decay=0.0)
    return {
        f"modular p=91, 1-layer transformer (full batch {n_mod})": (mod, n_mod),
        "MNIST MLP 784-200-200-10, 2000 points, batch 200, MSE": (mnist, 2000),
    }


def _synthetic_batch(cfg: ExperimentConfig, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    if cfg.task == "MODULAR":
        x = torch.randint(0, cfg.modulus, (n, 3), generator=g)
        x[:, 2] = cfg.modulus
        y = torch.randint(0, cfg.modulus, (n,), generator=g)
        return x.to(device).float(), y.to(device)
    return torch.rand(n, 1, 28, 28, generator=g).to(device), torch.randint(0, 10, (n,), generator=g).to(device)


def _time(fn, device, repeats: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _hvp_forward_over_reverse(ops: CurvatureOperators, V: torch.Tensor) -> torch.Tensor:
    """The alternative: torch.func jvp of grad, vmapped over probes, recomputing forward and backward each call."""
    grad_loss = torch.func.grad(ops.loss)
    return torch.func.vmap(lambda v: torch.func.jvp(grad_loss, (ops.theta,), (v,))[1])(V)


def _fresh_hvp_first_call(model, x, y, loss, V) -> torch.Tensor:
    """First HVP on a new operator, i.e. including building the cached first-order graph."""
    return CurvatureOperators(model, x, y, loss).hvp(V)


def bench(device: torch.device, repeats: int) -> list[str]:
    lines = []
    for name, (cfg, n_hessian) in _settings().items():
        torch.manual_seed(0)
        task = get_task(cfg)
        model = task.build_model(device)
        loss = lambda z, y: task.compute_training_loss(z, y, device)  # noqa: E731
        opt = OPTIMIZERS[cfg.optimizer](model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        x_train, y_train = _synthetic_batch(cfg, min(cfg.batch_size, n_hessian), device)
        x_h, y_h = _synthetic_batch(cfg, n_hessian, device)

        def train_step():
            opt.zero_grad(set_to_none=True)
            loss(model(x_train), y_train).backward()
            opt.step()

        t_step = _time(train_step, device, repeats * 5)

        model.eval()
        ops = CurvatureOperators(model, x_h, y_h, loss)
        V = rademacher_probes(cfg.spectral_k, ops.n_params, seed=0, device=device, dtype=ops.theta.dtype)
        ops.hvp(V)  # builds the cached first-order graph
        t_rr = _time(lambda: ops.hvp(V), device, repeats)
        t_rr_first = _time(lambda: _fresh_hvp_first_call(model, x_h, y_h, loss, V), device, repeats)
        t_fr = _time(lambda: _hvp_forward_over_reverse(ops, V), device, repeats)
        t_gn = _time(lambda: ops.gauss_newton(V), device, repeats)
        t_fn = _time(lambda: ops.functional(V), device, repeats)
        t_snap = _time(lambda: hessian_snapshot(model, x_h, y_h, loss, m=cfg.spectral_m, k=cfg.spectral_k,
                                                probe_seed=0), device, max(1, repeats // 2))

        lines += [
            f"### {name}",
            "",
            f"{ops.n_params:,} parameters; snapshot m={cfg.spectral_m}, k={cfg.spectral_k}; "
            f"Hessian batch {n_hessian} examples.",
            "",
            "| quantity | time | in training steps |",
            "|---|---|---|",
            f"| training step (batch {min(cfg.batch_size, n_hessian)}) | {t_step * 1e3:.2f} ms | 1 |",
            f"| batched HVP, k={cfg.spectral_k}, reverse-over-reverse on the cached graph (used) | {t_rr * 1e3:.2f} ms "
            f"(first call incl. building the graph: {t_rr_first * 1e3:.2f} ms) | {t_rr / t_step:.1f} |",
            f"| batched HVP, k={cfg.spectral_k}, torch.func forward-over-reverse (alternative) | {t_fr * 1e3:.2f} ms | {t_fr / t_step:.1f} |",
            f"| batched Gauss-Newton product G V, k={cfg.spectral_k} | {t_gn * 1e3:.2f} ms | {t_gn / t_step:.1f} |",
            f"| batched functional-part product H_f V, k={cfg.spectral_k} | {t_fn * 1e3:.2f} ms | {t_fn / t_step:.1f} |",
            f"| full snapshot | {t_snap * 1e3:.1f} ms | {t_snap / t_step:.0f} |",
            "",
            "| run length | snapshots (default schedule) | snapshot cost / training cost |",
            "|---|---|---|",
        ]
        for T in RUN_LENGTHS:
            n_snap = len(spectral_steps(T, growth=cfg.spectral_growth, every=cfg.spectral_every))
            lines.append(f"| {T:,} steps | {n_snap} (every {cfg.spectral_every}) | "
                         f"{n_snap * t_snap / (T * t_step):.2f}x |")
        lines.append("")
        model.train()
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("reports/hessian_benchmark.md"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hw = torch.cuda.get_device_name(device) if device.type == "cuda" else f"CPU ({platform.processor() or 'unknown'}), {torch.get_num_threads()} thread(s)"
    lines = [
        "# Hessian snapshot cost vs. training step",
        "",
        f"Generated by `python scripts/bench_hessian.py`. Device: {hw}; torch {torch.__version__}. "
        f"Medians over {args.repeats} repeats (training step: {args.repeats * 5}), after warm-up.",
        "",
    ] + bench(device, args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
