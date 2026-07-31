# training/loop.py

from __future__ import annotations

import os
import random
from dataclasses import fields
from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from analysis.eigenthings import estimate_density
from analysis.metrics import compute_accuracy, compute_loss
from analysis.spectral_observables import compute_spectral_observables_with_stderr

from configs.registry import OPTIMIZERS, ACTIVATIONS, LOSSES
from project_io.dir_making import make_dirs
from project_io.io_utils import save_checkpoint, save_training_data_npz, training_plot_stem
from project_io.graphing import save_log_spectral_snapshots, save_training_plot
from training.spectral_schedule import SpectralSchedule


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

    torch.set_default_dtype(cfg.dtype)
    set_seed(cfg.seed)

    device = _device()
    dirs = make_dirs(cfg, test_mode=cfg.test_mode)

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

    # Store spectral snapshots keyed by step
    t_grid_store: Dict[int, torch.Tensor] = {}
    phi_store: Dict[int, torch.Tensor] = {}

    # Fixed batch the Hessian loss is computed on at every spectral snapshot
    # in this run. Reusing the live training minibatch would let batch-to-
    # batch variation swamp the parameter-driven signal the spectrum is
    # supposed to track.
    spectral_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=min(cfg.spectral_batch_size, len(train_ds)),
        shuffle=False,
    )
    x_spectral, y_spectral = next(iter(spectral_loader))
    x_spectral = x_spectral.to(device).float()
    y_spectral = y_spectral.to(device)

    spectral_schedule = SpectralSchedule(
        growth=cfg.spectral_growth,
        dense_stride=cfg.spectral_dense_stride,
        gap_rate_threshold=cfg.spectral_gap_rate_threshold,
    )
    last_train_acc: float | None = None

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
                last_train_acc = tr_acc

                pbar.set_description(f"L: {tr_loss:1.1e}. A: {tr_acc*100:2.1f}%")

            # Evaluate
            if ((steps % cfg.eval_every == 0) and (steps > 0)) or (steps == 1):
                try:
                    test_loss = compute_loss(model, test_loader, cfg.loss_function, device, N=len(test_ds), dataset_name=None)
                    test_acc = compute_accuracy(model, test_loader, device, N=len(test_ds), dataset_name=None)

                    history["test_losses"].append(test_loss)
                    history["test_accuracies"].append(test_acc)
                    history["test_steps"].append(steps)

                    # Task-specific extra eval (e.g. MNIST's FGSM adversarial eval)
                    extra = task.extra_eval(model, steps)
                    for k, v in extra.items():
                        history.setdefault(k, []).append(_to_float(v))

                    # Transition-window detection needs both accuracies; skip
                    # until the first log_every tick has populated last_train_acc.
                    if last_train_acc is not None:
                        spectral_schedule.update_gap(steps, last_train_acc, test_acc)
                except Exception as e:
                    print(f"Warning: periodic eval failed at step {steps}: {e}")

            # Checkpoint: own cadence (checkpoint_every), own try/except.
            # Previously shared both with the eval block above, so a
            # checkpoint disk-write failure could silently skip the gap
            # update that drives the spectral schedule's grokking-transition
            # detection -- checkpointing is pure I/O and shouldn't be able to
            # take eval/spectral state down with it.
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

            # Spectral snapshot: geometric early, dense through the grokking
            # transition (see training/spectral_schedule.py).
            if cfg.run_spectral and spectral_schedule.should_log(steps):
                try:
                    model.eval()

                    # Fixed batch/probes: only the parameters differ between snapshots.
                    logits_spec = model(x_spectral)
                    loss_spec = task.compute_training_loss(logits_spec, y_spectral, device)

                    spec = estimate_density(
                        model=model,
                        m=cfg.spectral_m,
                        k=cfg.spectral_k,
                        sigma=cfg.spectral_sigma,
                        loss=loss_spec,
                        probe_seed=cfg.seed,
                    )

                    t_grid_store[steps] = spec.t_grid.detach().cpu()
                    phi_store[steps] = spec.density.detach().cpu()

                    obs, obs_stderr = compute_spectral_observables_with_stderr(
                        probe_nodes=spec.probe_nodes,
                        probe_weights=spec.probe_weights,
                        n_params=spec.n_params,
                    )

                    phi_max = spec.density.max().item()
                    is_null = phi_max < 1e-10

                    print(
                        f"Step {steps}: top_eig={obs.top_eig:.3e}±{obs_stderr.top_eig:.1e}, "
                        f"bulk_edge={obs.bulk_edge:.3e}, outliers={obs.outlier_count}, "
                        f"trace={obs.trace:.3e}±{obs_stderr.trace:.1e}, "
                        f"eff_rank={obs.effective_rank:.2f}, neg_mass={obs.negative_mass:.3e}, "
                        f"cond={obs.conditioning:.3e}"
                    )
                    if is_null:
                        print(f"WARNING: Spectrum appears null (max density={phi_max:.3e})")

                    for f in fields(obs):
                        history[f.name].append(_to_float(getattr(obs, f.name)))
                        history[f"{f.name}_stderr"].append(_to_float(getattr(obs_stderr, f.name)))
                    history["eig_steps"].append(steps)
                except Exception as e:
                    print(f"Warning: spectral snapshot failed at step {steps}: {e}")
                finally:
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
        phi_store=phi_store,
        t_store=t_grid_store,
    )

    save_training_data_npz(training_data_path, history=history)
    save_training_plot(plot_path, cfg=cfg, history=history)

    return history
