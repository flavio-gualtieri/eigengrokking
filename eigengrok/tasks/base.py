# eigengrok/tasks/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn

from eigengrok.config import ExperimentConfig
from eigengrok.registry import LOSSES


class Task(ABC):

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.history: Dict[str, list] = {
            # losses / accuracies
            "train_losses": [],
            "train_accuracies": [],

            "test_losses": [],
            "test_accuracies": [],

            # norms
            "norms": [],
            "last_layer_norms": [],

            # step counters
            "train_steps": [],
            "test_steps": [],
            "eig_steps": [],
        }
        # Hessian spectral statistics (eigengrok/hessian/spectrum.py) are added
        # by the training loop: one key per statistic plus a `<name>_stderr`.

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
        labels = labels.to(device)

        if self.cfg.loss_function == "CrossEntropy":
            return LOSSES["CrossEntropy"]()(logits, labels)
        elif self.cfg.loss_function == "MSE":
            one_hots = torch.eye(self.cfg.output_dim, device=device, dtype=logits.dtype)
            # Same value as nn.MSELoss() (mean over samples x classes), written
            # out because F.mse_loss's backward writes into the target's zero
            # tangent under torch.func forward-over-reverse, which the Hessian
            # operators use ("ZeroTensors are immutable", torch 2.14).
            return ((logits - one_hots[labels]) ** 2).mean()
        else:
            raise RuntimeError("Unreachable: loss_function validated earlier")

    def extra_eval(self, model: nn.Module, step: int) -> Dict[str, float]:
        """
        Optional per-task extra evaluation, run at eval_every cadence.
        Default: nothing.
        Returned keys are merged directly into self.history.
        """
        return {}

    def run(self) -> Dict[str, list]:
        """Delegate to the shared trainer. Do not put loop mechanics here."""
        from eigengrok.train.loop import run_experiment

        return run_experiment(self)
