from __future__ import annotations

import os
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import asdict
from typing import Dict, List, Optional, Any

import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from configs.config import ExperimentConfig
from configs.experiments import EXPERIMENTS
from data.data import get_torchvision_data
from configs.registry import OPTIMIZERS, ACTIVATIONS, LOSSES
from analysis.phi_eval_new import nth_moment, mass_above_thresh, pos_neg_ratio
from project_io.dir_making import make_dirs
from model.model import build_mlp, build_mlp_toy, set_seed
from analysis.eigenthings_new import estimate_density
from analysis.metrics import compute_accuracy, compute_loss
from model.adversarial import generate_fgsm_adversarial_examples
from pyhessian import hessian

from project_io.io_utils import save_checkpoint, save_training_data_npz, training_plot_stem

from project_io.graphing import save_log_spectral_snapshots, save_training_plot


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
        with torch.no_grad():
            total = sum((p.detach() ** 2).sum() for p in model.parameters())
            last = sum((p.detach() ** 2).sum() for p in model[-1].parameters())
        return float(torch.sqrt(total).item()), float(torch.sqrt(last).item())


def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_index",
        type=int,
        default=None,
        help="Run only a single experiment index from EXPERIMENTS",
    )
    return parser.parse_args()


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    # Set dtype and seed for reproducibility
    torch.set_default_dtype(cfg.dtype)
    set_seed(cfg.seed)

    # Set device and make directories
    device = _device()

    # Check if TOY run
    if os.environ.get("USE_TOY_MLP", "0") == "1":
        dirs = make_dirs(cfg, test_mode=cfg.test_mode, toy_mode=True)
    else:
        dirs = make_dirs(cfg, test_mode=cfg.test_mode)

    # Build MLP
    if os.environ.get("USE_TOY_MLP", "0") == "1":
        mlp = build_mlp_toy(
            activation=cfg.activation,
            initialization_scale=cfg.initialization_scale,
            device=device,
        )
    else:
        mlp = build_mlp(
            depth=cfg.depth,
            width=cfg.width,
            activation=cfg.activation,
            initialization_scale=cfg.initialization_scale,
            device=device,
    )

    # Enforce valid activation/optimizer/loss/dataset choice
    if cfg.activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation function:'{cfg.activation}'. Options: {list(ACTIVATIONS.keys())}")
    if cfg.optimizer not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'. Options: {list(OPTIMIZERS.keys())}")
    if cfg.loss_function not in LOSSES:
        raise ValueError(f"Unsupported loss '{cfg.loss_function}'. Options: {list(LOSSES.keys())}")
    
    # Retrieve functions form registries
    opt = OPTIMIZERS[cfg.optimizer](mlp.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = LOSSES[cfg.loss_function]()

    # Load data
    train_ds, test_ds, train_loader = get_torchvision_data(
        dataset=cfg.dataset,
        download_directory=cfg.download_directory,
        train_points=cfg.train_points,
        batch_size=cfg.batch_size,
    )

    # Create loaders for metrics
    train_eval_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Set number of classes and one-hots for loss computation
    num_classes = 10
    one_hots = torch.eye(num_classes, device=device, dtype=cfg.dtype)

    # Store spectral info
    t_grid_store: Dict[int, torch.Tensor] = {}
    phi_store: Dict[int, torch.Tensor] = {}

    # Initial adversatial set (before training starts)
    adv_examples, adv_labels = generate_fgsm_adversarial_examples(
        model=mlp,
        dataset=test_ds,
        epsilon=cfg.fgsm_epsilon,
        device=device,
        batch_size=cfg.batch_size,
    )
    adv_test_ds = torch.utils.data.TensorDataset(adv_examples, adv_labels)

    adv_test_loader = torch.utils.data.DataLoader(
        adv_test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # History dictionary to store training/test losses, accuracies, norms, etc. for plotting later
    history: Dict[str, List[float]] = {
        "train_losses": [],
        "train_accuracies": [],

        "test_losses": [],
        "test_accuracies": [],

        "adv_test_losses": [],
        "adv_test_accuracies": [],

        "norms": [],
        "last_layer_norms": [],

        "train_steps": [],
        "test_steps": [],
        "eig_steps": [],

        "eig_mass_gt1": [],
        "phi_pos_neg_ratio": [],
    
        "phi_1_moment": [],
        "phi_2_moment": [],
    }

    mlp.train()
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

            # Label check (REMOVE)
            if steps == 0:
                print("dim x:", x.shape, x.dtype, x.min().item(), x.max().item())
                print("dim y:", labels.shape, labels.dtype, labels.min().item(), labels.max().item())

            # Optimization step
            opt.zero_grad(set_to_none=True)
            x = x.to(device)
            x = x.float()

            logits = mlp(x)

            if cfg.loss_function == "CrossEntropy":
                loss = loss_fn(logits, labels.to(device))
            elif cfg.loss_function == "MSE":
                loss = loss_fn(logits, one_hots[labels])
            else:
                raise RuntimeError("Unreachable: loss_function validated earlier")
            
            loss.backward()
            opt.step()

            # Log metrics
            if ((steps % cfg.log_every == 0) and (steps > 0)) or (steps == 1):
                # Compute losses, accuracies, norms on train/test/adv_test and store in history
                tr_loss = compute_loss(mlp, train_eval_loader, cfg.loss_function, device, N=len(train_ds), dataset_name=cfg.dataset)
                tr_acc = compute_accuracy(mlp, train_eval_loader, device, N=len(train_ds), dataset_name=cfg.dataset)
                wn, lwn = _weight_norms(mlp)

                # Update history
                history["train_losses"].append(tr_loss)
                history["train_accuracies"].append(tr_acc)
                history["norms"].append(wn)
                history["last_layer_norms"].append(lwn)
                history["train_steps"].append(steps)

                pbar.set_description(f"L: {tr_loss:1.1e}. A: {tr_acc*100:2.1f}%")

                # Refresh adversarial set
                if steps == cfg.fgsm_refresh_step:
                    adv_examples, adv_labels = generate_fgsm_adversarial_examples(
                        model=mlp,
                        dataset=test_ds,
                        epsilon=cfg.fgsm_epsilon,
                        device=device,
                        batch_size=cfg.batch_size,
                    )
                    adv_test_ds = torch.utils.data.TensorDataset(adv_examples, adv_labels)

                    adv_test_loader = torch.utils.data.DataLoader(
                        adv_test_ds,
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=torch.cuda.is_available(),
                    )

            # Evaluate
            if ((steps % cfg.eval_every == 0) and (steps > 0)) or (steps == 1):
                try:
                    # Evaluate on clean and adversarial                
                    test_loss = compute_loss(mlp, test_loader, cfg.loss_function, device, N=len(test_ds), dataset_name=cfg.dataset)
                    test_acc = compute_accuracy(mlp, test_loader, device, N=len(test_ds), dataset_name=cfg.dataset)
    
                    adv_test_loss = compute_loss(mlp, adv_test_loader, cfg.loss_function, device, N=len(adv_test_ds), dataset_name=cfg.dataset)
                    adv_test_acc = compute_accuracy(mlp, adv_test_loader, device, N=len(adv_test_ds), dataset_name=cfg.dataset)

                    # Update history
                    history["test_losses"].append(test_loss)
                    history["test_accuracies"].append(test_acc)
                    history["adv_test_losses"].append(adv_test_loss)
                    history["adv_test_accuracies"].append(adv_test_acc)
                    history["test_steps"].append(steps)

                    # Checkpoint
                    ckpt_path = dirs["ckpt"] / (
                        f"mlp_checkpoint_step{steps}"
                        f"_depth{cfg.depth}_width{cfg.width}_scale{cfg.initialization_scale}.pt"
                    )
                    save_checkpoint(
                        path=ckpt_path,
                        step=steps,
                        model=mlp,
                        optimizer=opt,
                        history=history,
                        config=cfg,
                    )
                except Exception as e:
                    print(f"Warning: periodic eval/save failed at step {steps}: {e}")

            # Spectral snapshot
            if ((steps % cfg.spectral_every == 0) and (cfg.run_spectral == True) and (steps > 0)) or (steps == 1):
                try:
                    mlp.eval()

                    # recompute a fresh loss on the current batch / params
                    x_spec = x.detach()
                    y_spec = labels.to(device).detach()

                    logits_spec = mlp(x_spec)
                    if cfg.loss_function == "CrossEntropy":
                        loss_spec = loss_fn(logits_spec, y_spec)
                    elif cfg.loss_function == "MSE":
                        loss_spec = loss_fn(logits_spec, one_hots[y_spec])
                    else:
                        raise RuntimeError("Unreachable")

                    # Compute/store eigendensity and kappa
                    phi, t_grid = estimate_density(
                        model=mlp,
                        m=cfg.spectral_m,
                        k=cfg.spectral_k,
                        sigma=cfg.spectral_sigma,
                        loss = loss_spec
                        )
                    
                    # Store per-step spectral info
                    t_grid_store[steps] = t_grid.detach().cpu()
                    phi_store[steps] = phi.detach().cpu()
                    #kappa_store[steps] = float(kappa.detach().cpu().item()) if torch.is_tensor(kappa) else float(kappa)

                    # Run eval on phi
                    phi_mass_gt1 = mass_above_thresh(phi, t_grid, threshold=1.0)
                    phi_1_moment, phi_2_moment = nth_moment(phi, t_grid, n=1), nth_moment(phi, t_grid, n=2)
                    phi_pos_neg = pos_neg_ratio(phi, t_grid)
                    print(f"Step {steps}: mass λ>1: {phi_mass_gt1:.3e}, 1st moment: {phi_1_moment:.3e}, 2nd moment: {phi_2_moment:.3e}, pos-neg ratio: {phi_pos_neg:.3e}")

                    # Update history
                    history["eig_mass_gt1"].append(_to_float(phi_mass_gt1))
                    history["phi_1_moment"].append(_to_float(phi_1_moment))
                    history["phi_2_moment"].append(_to_float(phi_2_moment))
                    history["phi_pos_neg_ratio"].append(_to_float(phi_pos_neg))
                    history["eig_steps"].append(steps)
                except Exception as e:
                    print(f"Warning: spectral snapshot failed at step {steps}: {e}")

                finally:
                    mlp.train()

            steps += 1
            pbar.update(1)


    # Final saves: layer outputs and curves
    print("Final evaluation / saving")

    stem = f"depth{cfg.depth}_width{cfg.width}_scale{cfg.initialization_scale}"
    plot_stem = training_plot_stem(cfg)
    training_data_path = dirs["fig"] / f"training_data_{stem}.npz"
    plot_path = dirs["fig"] / f"training_curve_{plot_stem}.png"
    spectra_path = dirs["spectra"]

    save_log_spectral_snapshots(
        spectra_path=spectra_path,
        phi_store=phi_store,
        t_store=t_grid_store
    )

    print("train_steps:", len(history["train_steps"]))
    print("train_accuracies:", len(history["train_accuracies"]))
    print("test_steps:", len(history["test_steps"]))
    print("test_accuracies:", len(history["test_accuracies"]))
    print("adv_test_accuracies:", len(history["adv_test_accuracies"]))

    save_training_data_npz(training_data_path, history=history)
    save_training_plot(plot_path, cfg=cfg, history=history)

    return history


def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    cfg = EXPERIMENTS[task_id]
    run_experiment(cfg)


if __name__ == "__main__":
    main()

