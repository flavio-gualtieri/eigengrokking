from typing import Iterable

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector


def _flatten_grads(grads: Iterable[torch.Tensor], params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(g.reshape(-1))
    return torch.cat(parts)


def compute_full_hessian(
    loss: torch.Tensor,
    model: nn.Module,
) -> torch.Tensor:
    """
    Compute the exact full Hessian of `loss` with respect to all trainable
    parameters of `model`.

    Returns
    -------
    H : torch.Tensor
        Full Hessian matrix of shape [P, P], where P is the total number of
        trainable parameters.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("Model has no trainable parameters.")

    # First-order gradient, keeping graph for second derivatives
    first_grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    g = _flatten_grads(first_grads, params)   # shape [P]

    n_params = g.numel()
    H_rows: list[torch.Tensor] = []

    # One backward pass per gradient coordinate
    for i in range(n_params):
        second_grads = torch.autograd.grad(
            g[i],
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        H_rows.append(_flatten_grads(second_grads, params))

    H = torch.stack(H_rows, dim=0)  # [P, P]
    return H
