# tasks/mnist.py

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from tasks.base import Task
from models.mlp import MLP
from models.adversarial import generate_fgsm_adversarial_examples
from configs.base import ExperimentConfig
from data.torchvision_data import get_torchvision_data
from analysis.metrics import compute_accuracy, compute_loss

MNIST_INPUT_DIM = 784


class MNIST_task(Task):

    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg=cfg)
        self._test_ds = None
        self._adv_test_ds = None
        self._adv_test_loader = None
        self._adv_refreshed = False

    def build_model(self, device: torch.device) -> nn.Module:
        cfg = self.cfg
        if cfg.depth < 2:
            raise ValueError("This architecture requires depth >= 2")

        hidden_dims = (cfg.width,) * (cfg.depth - 1)

        return MLP(
            input_dim=MNIST_INPUT_DIM,
            hidden_dims=hidden_dims,
            output_dim=cfg.output_dim,
            init_scale=cfg.initialization_scale,
            activation_fn=cfg.activation,
            dropout=0.0,
            device=device,
        )

    def build_data(self) -> Dict[str, Any]:
        cfg = self.cfg

        train_ds, test_ds, train_loader = get_torchvision_data(
            dataset=cfg.task,
            download_directory=cfg.data_directory,
            train_points=cfg.train_points,
            batch_size=cfg.batch_size,
        )

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

        self._test_ds = test_ds

        return {
            "train_ds": train_ds,
            "test_ds": test_ds,
            "train_loader": train_loader,
            "train_eval_loader": train_eval_loader,
            "test_loader": test_loader,
        }

    def extra_eval(self, model: nn.Module, step: int) -> Dict[str, float]:
        cfg = self.cfg
        device = next(model.parameters()).device

        should_refresh = self._adv_test_loader is None or (
            not self._adv_refreshed and step >= cfg.fgsm_refresh_step
        )

        if should_refresh:
            adv_examples, adv_labels = generate_fgsm_adversarial_examples(
                model=model,
                dataset=self._test_ds,
                epsilon=cfg.fgsm_epsilon,
                device=device,
                batch_size=cfg.batch_size,
            )
            self._adv_test_ds = torch.utils.data.TensorDataset(adv_examples, adv_labels)
            self._adv_test_loader = torch.utils.data.DataLoader(
                self._adv_test_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            if step >= cfg.fgsm_refresh_step:
                self._adv_refreshed = True

        adv_test_loss = compute_loss(
            model, self._adv_test_loader, cfg.loss_function, device,
            N=len(self._adv_test_ds), dataset_name=None,
        )
        adv_test_acc = compute_accuracy(
            model, self._adv_test_loader, device,
            N=len(self._adv_test_ds), dataset_name=None,
        )

        return {
            "adv_test_losses": adv_test_loss,
            "adv_test_accuracies": adv_test_acc,
        }
