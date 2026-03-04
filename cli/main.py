from __future__ import annotations

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
from analysis.phi_eval import phi_mass_g1, phi_mean_var
from project_io.dir_making import make_dirs
from model.model import build_mlp, set_seed
from analysis.eigenthings import lanczos_phi_average
from analysis.metrics import compute_accuracy, compute_loss
from model.adversarial import generate_fgsm_adversarial_examples

from project_io.io_utils import (
    save_checkpoint,
    save_log_spectral_snapshots,
    save_training_data_npz,
    save_training_plot_png,
    save_spectral_snapshots,
    training_plot_stem,
)


def _device(cfg: ExperimentConfig) -> torch.device:
    """
    Sets devide to MPS if available (otherwise CPU)
    """
    if cfg.hpc == True:
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def _weight_norms(model: torch.nn.Module) -> tuple[float, float]:
        with torch.no_grad():
            total = sum((p.detach() ** 2).sum() for p in model.parameters())
            last = sum((p.detach() ** 2).sum() for p in model[-1].parameters())
        return float(torch.sqrt(total).item()), float(torch.sqrt(last).item())


def _should_log(cfg: ExperimentConfig, step: int) -> bool:
        return (step < 30) or (step < 150 and step % 10 == 0) or (step % cfg.log_freq == 0)


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
    device = _device(cfg)
    dirs = make_dirs(cfg, test_mode=cfg.test_mode)

    # Build MLP
    mlp = build_mlp(
        depth=cfg.depth,
        width=cfg.width,
        activation=cfg.activation,
        initialization_scale=cfg.initialization_scale,
        device=device,
        dataset=cfg.dataset,
        modulus=cfg.modulus if cfg.dataset == "MODULAR" else None,
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
        p = cfg.modulus if cfg.dataset == "MODULAR" else None,
    )

    # Set number of classes and one-hots for loss computation
    num_classes = cfg.modulus if cfg.dataset == "MODULAR" else 10
    one_hots = torch.eye(num_classes, device=device, dtype=cfg.dtype)

    # Grid for spectral density estimation CHANGE TO DYNAMIC BASED ON SPECTRAL RANGE OBSERVED
    # Store per-step t grids to avoid shape mismatches when t changes over time.
    t_store: Dict[int, torch.Tensor] = {}

    # Init storage for spectral snapshots (phi, kappa, eig_mass_gt1)
    phi_store: Dict[int, torch.Tensor] = {}
    kappa_store: Dict[int, float] = {}
    eig_mass_gt1_store: Dict[int, float] = {}

    # Probe batch for spectral analysis (fixed throughout training)
    # Use train_ds since it has labels (for Lanczos)
    # If dataset is MNIST, shape to [B, 784] and convert to float
    # If MODULAR, keep as is (expect shape [B, 2p])
    # STUDY PIN MEMORY. GPU TRICKERY HERE
    probe_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=min(cfg.spectral_probe_batch_size, len(train_ds)),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
    )
    x_probe, y_probe = next(iter(probe_loader))
    x_probe = x_probe.to(device)
    y_probe = y_probe.to(device)

    if cfg.dataset == "MODULAR":
        X_probe = x_probe  # expect shape [B, 2p]
        print("X_probe shape:", X_probe.shape)
        print("y_probe shape:", y_probe.shape)
    else:
        X_probe = x_probe.view(x_probe.size(0), -1)  # MNIST [B, 784]

    # Initial adversatial set (before training starts)
    if cfg.dataset in ["MNIST", "FashionMNIST"]:
        adv_examples, adv_labels = generate_fgsm_adversarial_examples(
            model=mlp,
            dataset=test_ds,
            epsilon=cfg.fgsm_epsilon,
            device=device,
            batch_size=cfg.batch_size,
        )
        adv_test_ds = torch.utils.data.TensorDataset(adv_examples, adv_labels)
    else:
        adv_test_ds = None

    # History dictionary to store training/test losses, accuracies, norms, etc. for plotting later
    history: Dict[str, List[float]] = {
        "train_losses": [],
        "test_losses": [],
        "adv_test_losses": [],
        "train_accuracies": [],
        "test_accuracies": [],
        "adv_test_accuracies": [],
        "norms": [],
        "last_layer_norms": [],
        "train_log_steps": [],
        "test_log_steps": [],
        "eig_mass_gt1": [],
        "eig_log_steps": [],
        "kappa": [],
        "phi_mean": [],
        "phi_var": [],
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
                print("x:", x.shape, x.dtype, x.min().item(), x.max().item())
                print("labels:", labels.shape, labels.dtype, labels.min().item(), labels.max().item())

            # Log metrics
            if _should_log(cfg, steps):
                # Compute losses, accuracies, norms on train/test/adv_test and store in history
                tr_loss = compute_loss(mlp, train_ds, cfg.loss_function, device, N=len(train_ds), dataset_name=cfg.dataset)
                tr_acc = compute_accuracy(mlp, train_ds, device, N=len(train_ds), dataset_name=cfg.dataset)
                wn, lwn = _weight_norms(mlp)

                # Update history
                history["train_losses"].append(tr_loss)
                history["train_accuracies"].append(tr_acc)
                history["norms"].append(wn)
                history["last_layer_norms"].append(lwn)
                history["train_log_steps"].append(steps)

                pbar.set_description(f"L: {tr_loss:1.1e}. A: {tr_acc*100:2.1f}%")

            # Periodic saving and evaluation
            if (steps % cfg.save_every == 0) and steps > 0:
                # Refresh adversarial set
                if cfg.dataset in ["MNIST", "FashionMNIST"] and steps == cfg.fgsm_refresh_step:
                    adv_examples, adv_labels = generate_fgsm_adversarial_examples(
                        model=mlp,
                        dataset=test_ds,
                        epsilon=cfg.fgsm_epsilon,
                        device=device,
                        batch_size=cfg.batch_size,
                    )
                    adv_test_ds = torch.utils.data.TensorDataset(adv_examples, adv_labels)

                try:
                    # Evaluate on clean and adversarial
                    adv_loss = float("nan")
                    adv_acc = float("nan")
            
                    te_loss = compute_loss(mlp, test_ds, cfg.loss_function, device, N=len(test_ds), dataset_name=cfg.dataset)
                    te_acc = compute_accuracy(mlp, test_ds, device, N=len(test_ds), dataset_name=cfg.dataset)
    
                    if cfg.dataset in ["MNIST", "FashionMNIST"]:
                        adv_loss = compute_loss(mlp, adv_test_ds, cfg.loss_function, device, N=len(adv_test_ds), dataset_name=cfg.dataset)
                        adv_acc = compute_accuracy(mlp, adv_test_ds, device, N=len(adv_test_ds), dataset_name=cfg.dataset)

                    # Update history
                    history["test_losses"].append(te_loss)
                    history["test_accuracies"].append(te_acc)
                    if cfg.dataset in ["MNIST", "FashionMNIST"]:
                        history["adv_test_losses"].append(adv_loss)
                        history["adv_test_accuracies"].append(adv_acc)
                    history["test_log_steps"].append(steps)

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

            if steps % cfg.spectral_every == 0 and cfg.run_spectral == True:
                # Spectral snapshot
                try:
                    # Set to eval mode for spectral analysis
                    mlp.eval()

                    # Compute/store eigendensity and kappa
                    phi, kappa, t_dyn = lanczos_phi_average(
                        model=mlp,
                        X=X_probe,
                        y=y_probe,
                        m=cfg.spectral_m,
                        k=cfg.spectral_k,
                        sigma=cfg.spectral_sigma
                        )
                    
                    # Store per-step t grid for correct plotting
                    t_store[steps] = t_dyn.detach().cpu()

                    phi_store[steps] = phi.detach().cpu()
                    kappa_store[steps] = float(kappa.detach().cpu().item()) if torch.is_tensor(kappa) else float(kappa)

                    # Compute mass > 1
                    mass_fraction_gt1 = phi_mass_g1(phi, t_dyn)
                    phi_mean, phi_var = phi_mean_var(phi, t_dyn)

                    # Update history and store mass fraction > 1
                    eig_mass_gt1_store[steps] = mass_fraction_gt1
                    history["eig_mass_gt1"].append(mass_fraction_gt1)
                    history["eig_log_steps"].append(steps)
                    history["kappa"].append(kappa)
                    history["phi_mean"].append(phi_mean)
                    history["phi_var"].append(phi_var)

                finally:
                    mlp.train()

            # Optimization step
            opt.zero_grad(set_to_none=True)
            x = x.to(device)
            if cfg.dataset != "MODULAR":
                # MNIST: x is [B, 1, 28, 28], model has Flatten anyway, but keep consistent with probe shaping
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

            steps += 1
            pbar.update(1)


    # Final saves: layer outputs and curves
    print("Final evaluation / saving")

    save_log_spectral_snapshots(
        spectra_dir=dirs["spectra"],
        phi_store=phi_store,
        t_store=t_store,
        dpi=200,
    )

    stem = f"depth{cfg.depth}_width{cfg.width}_scale{cfg.initialization_scale}"
    plot_stem = training_plot_stem(cfg)

    training_data_path = dirs["fig"] / f"training_data_{stem}.npz"
    save_training_data_npz(training_data_path, history=history)

    plot_path = dirs["fig"] / f"training_curve_{plot_stem}.png"
    save_training_plot_png(plot_path, cfg=cfg, history=history)

    print(f"Saved training data: {training_data_path}")
    print(f"Saved training plot: {plot_path}")

    return history

""" 
def main() -> None:
    args = parse_args()

    if args.exp_index is None:
        # original behaviour (local runs unchanged)
        for cfg in EXPERIMENTS:
            run_experiment(cfg)
    else:
        idx = args.exp_index

        if idx < 0 or idx >= len(EXPERIMENTS):
            raise ValueError(
                f"exp_index {idx} out of range (0-{len(EXPERIMENTS)-1})"
            )

        run_experiment(EXPERIMENTS[idx]) """

def main() -> None:
    for cfg in EXPERIMENTS:
        run_experiment(cfg)


if __name__ == "__main__":
    main()
