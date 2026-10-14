# eigengrok/hessian/operators.py

"""
Matrix-free curvature operators for a model's loss on a fixed batch.

Every operator maps a batch of flat parameter-space vectors V, shape [k, P],
to [k, P]; the k probes go through one vmapped call instead of a Python loop.

    hvp(V)           H V     H   = d^2 L / d theta^2                  (full Hessian)
    gauss_newton(V)  G V     G   = J^T H_z J                          (Gauss-Newton)
    functional(V)    H_f V   H_f = sum_{n,c} (dL/dz_nc) d^2 z_nc / d theta^2

with z = f(theta) the logits, J = dz/dtheta and H_z = d^2 L / dz^2. For any
loss, H = G + H_f exactly, and G is positive semi-definite whenever L is convex
in z (cross-entropy, MSE). H_f is computed on its own, as the Hessian of
<stopgrad(dL/dz), f(theta)>, not as H - G, so that identity is a real check
(tests/test_hessian.py).

H V is double backprop through a first-order gradient graph that is built
once, on the first call, and reused by every later call (one batched backward
pass per call, probes batched with is_grads_batched). On an A100 this was
1.2x (transformer) to 1.8x (MLP) faster than torch.func forward-over-reverse,
which recomputes the forward and backward pass on every call
(reports/hessian_benchmark.md). G V and H_f V use torch.func.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.func import functional_call, grad, jvp, vjp, vmap

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class CurvatureOperators:
    """
    Curvature of `loss_fn(model(inputs), targets)` at the model's current
    parameters, as a function of the flat vector of its trainable parameters.

    `chunk_size` caps how many probe vectors are vmapped at once (memory vs.
    speed); None pushes all k through together.
    """

    def __init__(
            self,
            model: nn.Module,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            loss_fn: LossFn,
            *,
            chunk_size: int | None = None,
    ):
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.loss_fn = loss_fn
        self.chunk_size = chunk_size

        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self._names = [name for name, _ in trainable]
        self._shapes = [p.shape for _, p in trainable]
        self._numels = [p.numel() for _, p in trainable]
        self._fixed = {name: p.detach() for name, p in model.named_parameters() if not p.requires_grad}
        self._fixed.update(dict(model.named_buffers()))

        self.theta = torch.cat([p.detach().reshape(-1) for _, p in trainable])
        self.n_params = self.theta.numel()
        self._grad_graph: tuple[torch.Tensor, torch.Tensor] | None = None

    # --- the loss as a function of the flat parameter vector ---

    def unflatten(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        chunks = flat.split(self._numels)
        return {name: c.view(shape) for name, c, shape in zip(self._names, chunks, self._shapes)}

    def logits(self, flat: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, {**self.unflatten(flat), **self._fixed}, (self.inputs,))

    def loss(self, flat: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.logits(flat), self.targets)

    def _loss_of_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(z, self.targets)

    # --- operators ---

    def hvp(self, V: torch.Tensor) -> torch.Tensor:
        """H V for V of shape [k, P]."""
        if self._grad_graph is None:
            theta = self.theta.clone().requires_grad_(True)
            g = torch.autograd.grad(self.loss(theta), theta, create_graph=True)[0]
            self._grad_graph = (theta, g)
        theta, g = self._grad_graph
        chunks = V.split(self.chunk_size or len(V))
        return torch.cat([
            torch.autograd.grad(g, theta, grad_outputs=chunk, is_grads_batched=True, retain_graph=True)[0]
            for chunk in chunks
        ])

    def gauss_newton(self, V: torch.Tensor) -> torch.Tensor:
        """G V = J^T (H_z (J V)) for V of shape [k, P]."""
        z0, pullback = vjp(self.logits, self.theta)
        grad_z = grad(self._loss_of_logits)

        def one(v):
            jv = jvp(self.logits, (self.theta,), (v,))[1]
            h_jv = jvp(grad_z, (z0,), (jv,))[1]
            return pullback(h_jv)[0]

        return vmap(one, chunk_size=self.chunk_size)(V)

    def functional(self, V: torch.Tensor) -> torch.Tensor:
        """H_f V, the Hessian of <stopgrad(dL/dz), f(theta)>, for V of shape [k, P]."""
        dl_dz = grad(self._loss_of_logits)(self.logits(self.theta)).detach()

        def residual_weighted_logits(flat):
            return (dl_dz * self.logits(flat)).sum()

        grad_r = grad(residual_weighted_logits)

        def one(v):
            return jvp(grad_r, (self.theta,), (v,))[1]

        return vmap(one, chunk_size=self.chunk_size)(V)
