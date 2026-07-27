# tasks/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, Dict

import torch
import torch.nn as nn

from configs.base import ExperimentConfig
from configs.registry import LOSSES
from analysis.spectral_observables import SpectralObservables


class Task(ABC):

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.history: Dict[str, list] = {
            # losses / accuracies
            "train_losses": [],
            "train_accuracies": [],

            "test_losses": [],
            "test_accuracies": [],

            "adv_test_losses": [],
            "adv_test_accuracies": [],

            # norms
            "norms": [],
            "last_layer_norms": [],

            # step counters
            "train_steps": [],
            "test_steps": [],
            "eig_steps": [],
        }

        # curvature / spectral observables: point estimate (pooled across
        # probes) plus its probe-wise standard error, one pair per field of
        # SpectralObservables -- see analysis/spectral_observables.py.
        for f in fields(SpectralObservables):
            self.history[f.name] = []
            self.history[f"{f.name}_stderr"] = []

    @abstractmethod
    def build_model(self, device: torch.device) -> nn.Module:
        """Construct and return the model for this task, placed on `device`."""
        ...

    @abstractmethod
    def build_data(self) -> Dict[str, Any]:
        """
        Build and return a dict with at least the keys:
            "train_ds", "test_ds", "train_loader", "train_eval_loader", "test_loader"
        """
        ...

    def compute_training_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generic per-step training loss, driven by cfg.loss_function.
        CrossEntropy uses raw integer labels; MSE compares against one-hots
        sized by cfg.output_dim. Shared across tasks so nobody has to
        reimplement this branching.
        """
        loss_fn = LOSSES[self.cfg.loss_function]()
        labels = labels.to(device)

        if self.cfg.loss_function == "CrossEntropy":
            return loss_fn(logits, labels)
        elif self.cfg.loss_function == "MSE":
            one_hots = torch.eye(self.cfg.output_dim, device=device, dtype=logits.dtype)
            return loss_fn(logits, one_hots[labels])
        else:
            raise RuntimeError("Unreachable: loss_function validated earlier")

    def extra_eval(self, model: nn.Module, step: int) -> Dict[str, float]:
        """
        Optional per-task extra evaluation, run at eval_every cadence.
        Default: nothing. MNIST overrides this for FGSM adversarial eval.
        Returned keys are merged directly into self.history.
        """
        return {}

    def run(self) -> Dict[str, list]:
        """Delegate to the shared trainer. Do not put loop mechanics here."""
        from training.loop import run_experiment

        return run_experiment(self)
