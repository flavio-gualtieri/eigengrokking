from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn

from configs.registry import ACTIVATIONS



def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(
        *,
        depth: int,
        width: int,
        activation: str,
        initialization_scale: float,
        device: torch.device,
        input_dim: int = 784,
        output_dim: int = 10
        ) -> nn.Sequential:
    
    # Enforce depth and activation choice
    if depth < 2:
        raise ValueError("This architecture requires depth >= 2")

    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation function:'{activation}'. Options: {list(ACTIVATIONS.keys())}")
    
    # Retrieve activation from registry
    activation_fn = ACTIVATIONS[activation]

    # Initialize layers list with input flattening
    layers: list[nn.Module] = [nn.Flatten()]

    # Build hidden layers
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(input_dim, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, output_dim))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())

    mlp = nn.Sequential(*layers).to(device)

    with torch.no_grad():
        for p in mlp.parameters():
            p.data = initialization_scale * p.data

    return mlp


def build_mlp_toy(
        *,
        activation: str,
        initialization_scale: float,
        device: torch.device,
        input_dim: int = 784,
        output_dim: int = 10
        ) -> nn.Sequential:
    
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation function: '{activation}'. "
            f"Options: {list(ACTIVATIONS.keys())}"
        )

    activation_fn = ACTIVATIONS[activation]

    layers: list[nn.Module] = [
        nn.Flatten(),
        nn.Linear(input_dim, 2),
        activation_fn(),
        nn.Linear(2, 2),
        activation_fn(),
        nn.Linear(2, output_dim),
    ]

    mlp = nn.Sequential(*layers).to(device)

    with torch.no_grad():
        for p in mlp.parameters():
            p.data = initialization_scale * p.data

    return mlp
