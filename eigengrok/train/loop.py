# eigengrok/train/loop.py

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from eigengrok.train.metrics import compute_accuracy, compute_loss
from eigengrok.hessian.snapshot import fixed_eval_batch, hessian_snapshot

from eigengrok.registry import OPTIMIZERS, ACTIVATIONS, LOSSES
from eigengrok.outputs import make_dirs
from eigengrok.outputs import save_checkpoint, save_training_data_npz, training_plot_stem
from eigengrok.plotting.training_curves import save_log_spectral_snapshots, save_training_plot
from eigengrok.train.spectral_schedule import spectral_steps


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")
    else:
        print("Using CPU device")
        return torch.device("cpu")


def _weight_norms(model: torch.nn.Module) -> tuple[float, float]:
    """Total parameter norm, and the norm of the last Linear layer found in
    the model. Generic over architecture (doesn't assume nn.Sequential)."""
    with torch.no_grad():
        total = sum((p.detach() ** 2).sum() for p in model.parameters())
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        if linears:
            last = sum((p.detach() ** 2).sum() for p in linears[-1].parameters())
        else:
            last = total
    return float(torch.sqrt(total).item()), float(torch.sqrt(last).item())


def _to_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def run_experiment(task) -> Dict[str, Any]:
    """
    Generic training loop, shared by every Task. Contains zero task-specific
    logic -- all of that is delegated to `task` (build_model, build_data,
    compute_training_loss, extra_eval).
    """
    cfg = task.cfg

    torch.set_default_dtype(cfg.torch_dtype)
    set_seed(cfg.seed)

    device = _device()
    dirs = make_dirs(cfg)

    model = task.build_model(device)

    if cfg.activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation function: '{cfg.activation}'. Options: {list(ACTIVATIONS.keys())}")
    if cfg.optimizer not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'. Options: {list(OPTIMIZERS.keys())}")
    if cfg.loss_function not in LOSSES:
        raise ValueError(f"Unsupported loss '{cfg.loss_function}'. Options: {list(LOSSES.keys())}")

    opt = OPTIMIZERS[cfg.optimizer](model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    data = task.build_data()
    train_loader = data["train_loader"]
    train_eval_loader = data["train_eval_loader"]
    test_loader = data["test_loader"]
    train_ds = data["train_ds"]
    test_ds = data["test_ds"]

    history = task.history

    # Spectral densities (for plotting) keyed by step
    grid_store: Dict[int, np.ndarray] = {}
    density_store: Dict[int, np.ndarray] = {}

    # The Hessian is always of the loss on the same fixed batch, so
    # successive snapshots differ only through the parameters.
    x_hessian, y_hessian = fixed_eval_batch(train_ds, cfg.spectral_max_points, seed=cfg.seed, device=device)
    snapshot_steps = set(spectral_steps(cfg.optimization_steps, growth=cfg.spectral_growth, every=cfg.spectral_every))
    hessian_loss = lambda logits, labels: task.compute_training_loss(logits, labels, device)  # noqa: E731

    model.train()
    steps = 0

    print("Training")
    with tqdm(total=cfg.optimization_steps) as pbar:
        loader_iter = iter(train_loader)

        while steps < cfg.optimization_steps:
            # Fetch next batch (refreshing iterator if end of epoch reached)
            try:
                x, labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                x, labels = next(loader_iter)

            # Optimization step
            opt.zero_grad(set_to_none=True)
            x = x.to(device).float()

            logits = model(x)
            loss = task.compute_training_loss(logits, labels, device)

            loss.backward()
            opt.step()

            # Log metrics
            if ((steps % cfg.log_every == 0) and (steps > 0)) or (steps == 1):
                tr_loss = compute_loss(model, train_eval_loader, cfg.loss_function, device, N=len(train_ds), dataset_name=None)
                tr_acc = compute_accuracy(model, train_eval_loader, device, N=len(train_ds), dataset_name=None)
                wn, lwn = _weight_norms(model)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
                history.setdefault("grad_norms", []).append(grad_norm)

                history["train_losses"].append(tr_loss)
                history["train_accuracies"].append(tr_acc)
                history["norms"].append(wn)
                history["last_layer_norms"].append(lwn)
                history["train_steps"].append(steps)

                pbar.set_description(f"L: {tr_loss:1.1e}. A: {tr_acc*100:2.1f}%")

            # Evaluate
            if ((steps % cfg.eval_every == 0) and (steps > 0)) or (steps == 1):
                try:
                    test_loss = compute_loss(model, test_loader, cfg.loss_function, device, N=len(test_ds), dataset_name=None)
                    test_acc = compute_accuracy(model, test_loader, device, N=len(test_ds), dataset_name=None)

                    history["test_losses"].append(test_loss)
                    history["test_accuracies"].append(test_acc)
                    history["test_steps"].append(steps)

                    # Task-specific extra eval (none of the current tasks define one)
                    extra = task.extra_eval(model, steps)
                    for k, v in extra.items():
                        history.setdefault(k, []).append(_to_float(v))
                except Exception as e:
                    print(f"Warning: periodic eval failed at step {steps}: {e}")

            # Checkpoint: own cadence (checkpoint_every), own try/except.
            # Previously shared both with the eval block above, so a
            # checkpoint disk-write failure could silently skip eval -- checkpointing
            # is pure I/O and shouldn't be able to take eval state down with it.
            if ((steps % cfg.checkpoint_every == 0) and (steps > 0)) or (steps == 1):
                try:
                    ckpt_path = dirs["ckpt"] / f"checkpoint_step{steps}.pt"
                    save_checkpoint(
                        path=ckpt_path,
                        step=steps,
                        model=model,
                        optimizer=opt,
                        history=history,
                        config=cfg,
                    )
                except Exception as e:
                    print(f"Warning: checkpoint save failed at step {steps}: {e}")

            # Hessian spectral snapshot, on a fixed, label-blind schedule
            # (eigengrok/train/spectral_schedule.py).
            if cfg.run_spectral and steps in snapshot_steps:
                try:
                    model.eval()
                    snap = hessian_snapshot(
                        model, x_hessian, y_hessian, hessian_loss,
                        m=cfg.spectral_m, k=cfg.spectral_k, probe_seed=cfg.seed,
                        sigma_frac=cfg.spectral_sigma_frac,
                    )
                    grid_store[steps] = snap.grid
                    density_store[steps] = snap.density
                    for name, value in snap.stats.items():
                        history.setdefault(name, []).append(value)
                        history.setdefault(f"{name}_stderr", []).append(snap.stderr[name])
                    history["eig_steps"].append(steps)

                    st = snap.stats
                    print(
                        f"Step {steps}: top_eig={st['top_eig']:.3e}, trace={st['trace']:.3e}"
                        f"±{snap.stderr['trace']:.1e}, eff_rank={st['effective_rank']:.2f}, "
                        f"neg_mass={st['negative_mass']:.3e}, n_outliers={st['n_outliers']:.1f}"
                    )
                finally:  # no except: a failed snapshot must stop the run, not leave a silent gap
                    model.train()

            steps += 1
            pbar.update(1)

    # Final saves: spectral snapshots and training curves
    print("Final evaluation / saving")

    plot_stem = training_plot_stem(cfg)
    training_data_path = dirs["fig"] / f"training_data_{plot_stem}.npz"
    plot_path = dirs["fig"] / f"training_curve_{plot_stem}.png"
    spectra_path = dirs["spectra"]

    save_log_spectral_snapshots(
        spectra_path=spectra_path,
        phi_store=density_store,
        t_store=grid_store,
    )

    save_training_data_npz(training_data_path, history=history)
    save_training_plot(plot_path, cfg=cfg, history=history)
    print(f"Outputs written to {dirs['base']}")

    return history
