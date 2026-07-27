# tasks/mod_add.py

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from tasks.base import Task
from models.transformer import Transformer
from data.mod_add import get_modular_addition_data, N_CTX


class ModularAddition_task(Task):

    def build_model(self, device: torch.device) -> nn.Module:
        cfg = self.cfg
        if cfg.depth < 1:
            raise ValueError("This architecture requires depth >= 1")

        vocab_size = cfg.modulus + 1  # numbers [0, modulus) plus the "=" token

        return Transformer(
            vocab_size=vocab_size,
            output_dim=cfg.modulus,
            n_ctx=N_CTX,
            d_model=cfg.width,
            num_layers=cfg.depth,
            num_heads=cfg.num_heads,
            activation_fn=cfg.activation,
            init_scale=cfg.initialization_scale,
            device=device,
        )

    def build_data(self) -> Dict[str, Any]:
        cfg = self.cfg

        train_ds, test_ds, train_loader = get_modular_addition_data(
            modulus=cfg.modulus,
            train_frac=cfg.train_frac,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
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
