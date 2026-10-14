# eigengrok/tasks/mnist.py

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from eigengrok.tasks.base import Task
from eigengrok.models.mlp import MLP
from eigengrok.data.vision import get_torchvision_data

MNIST_INPUT_DIM = 784


class MNIST_task(Task):

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

        return {
            "train_ds": train_ds,
            "test_ds": test_ds,
            "train_loader": train_loader,
            "train_eval_loader": train_eval_loader,
            "test_loader": test_loader,
        }
