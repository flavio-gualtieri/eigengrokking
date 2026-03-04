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
        dataset: str = "MNIST",
        modulus: int | None = None,
        input_dim: int = 784,
        output_dim: int = 10
        ) -> nn.Sequential:
    
    # Enforce depth and activation choice
    if depth < 2:
        raise ValueError("This architecture requires depth >= 2")

    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation function:'{activation}'. Options: {list(ACTIVATIONS.keys())}")
    
    # Adjust dimensions for modular arithmetic dataset
    if dataset == "MODULAR":
        if modulus is None:
            raise ValueError("For MODULAR dataset, 'modulus' must be provided.")

        input_dim = 2 * modulus
        output_dim = modulus
    
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