# models/mlp.py

from __future__ import annotations

import torch
import torch.nn as nn

from configs.registry import ACTIVATIONS


class MLP(nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            output_dim: int,
            device: torch.device,
            init_scale: float = 1.0,
            activation_fn: str = "ReLU",
            dropout: float = 0.0,
    ):
        super().__init__()
        activation_cls = ACTIVATIONS[activation_fn]
        layers: list[nn.Module] = [nn.Flatten()]
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation_cls(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers).to(device)

        with torch.no_grad():
            for p in self.mlp.parameters():
                p.data = init_scale * p.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
