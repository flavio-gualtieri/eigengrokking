from __future__ import annotations
from collections.abc import Iterable

import math
import torch
from torch import device, nn
from torchgen import model


class Eigenthings:
    def __init__(
              self,
              model: nn.Module,
              m: int,
              k: int,
              t_grid: torch.Tensor,
              sigma: float,
              params: list[torch.Tensor],
              device: torch.device,
              loss
    ):
        self.model = model
        self.m = m
        self.k = k
        self.t_grid = t_grid
        self.sigma = sigma
        self.loss = loss

        self.device = device
        self.params = params


    def gaussian_kernel(self, t, center, sigma):
        coeff = sigma * math.sqrt(2.0 * math.pi)
        return torch.exp(-0.5 * ((t - center) / sigma) ** 2) / coeff
        

    def unpack_vec(self, v: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        i = 0
        for p in self.params:
            n = p.numel()
            out.append(v[i : i + n].view_as(p))
            i += n

        return out


    def pack_list(self, xs: Iterable[torch.Tensor]) -> torch.Tensor:
        xs = list(xs)
        if len(xs) == 0:
            return torch.empty(0, device=self.device)
        
        return torch.cat([x.reshape(-1) for x in xs], dim=0)


    def hvp(self, q):
        q = q.to(self.device)
        q_list = self.unpack_vec(q)

        grads = torch.autograd.grad(self.loss, self.params, create_graph=True)

        gv = torch.zeros((), device=self.device, dtype=self.loss.dtype)
        for g, v in zip(grads, q_list):
            gv = gv + (g * v).sum()

        Hv = torch.autograd.grad(gv, self.params, retain_graph=True, create_graph=False)

        return self.pack_list(Hv)


    def lanczos(self, v0):
        v0 = torch.as_tensor(v0, dtype=self.params[0].dtype, device=self.device)
        n = v0.shape[0]

        q_prev = torch.zeros(n, dtype=v0.dtype, device=self.device)
        q = v0 / torch.linalg.norm(v0)

        alpha = []
        beta = []
        Q = [q.clone()]

        beta_prev = torch.tensor(0.0, dtype=v0.dtype, device=self.device)

        for _ in range(self.m):
            w = self.hvp(q) - beta_prev * q_prev
            a = torch.dot(q, w)
            w = w - a * q

            for qi in Q:
                w = w - torch.dot(qi, w) * qi

            b = torch.linalg.norm(w)

            alpha.append(a)

            if b < 1e-12:
                break

            beta.append(b)
            q_prev = q
            q = w / b
            Q.append(q.clone())
            beta_prev = b

        k = len(alpha)
        T = torch.zeros((k, k), dtype=v0.dtype, device=self.device)
        for i in range(k):
            T[i, i] = alpha[i]
            if i < k - 1:
                T[i, i + 1] = beta[i]
                T[i + 1, i] = beta[i]

        Q = torch.stack(Q[:k], dim=1)

        alpha = torch.stack(alpha)
        beta = torch.stack(beta) if beta else torch.empty(0, dtype=v0.dtype, device=self.device)

        return T


    def lanczos_quadrature(self, v0):
        T = self.lanczos(v0)

        T_cpu = T.to("cpu")
        eigvals, eigvecs = torch.linalg.eigh(T_cpu)

        nodes = eigvals.to(self.device)
        weights = (eigvecs[0, :] ** 2).to(self.device)

        return nodes, weights


    def density_from_lanczos(self, nodes, weights, t_grid, sigma):
        density = torch.zeros_like(t_grid, dtype=torch.float32)
        for lmbda, w in zip(nodes, weights):
            density += w * self.gaussian_kernel(t_grid, lmbda, sigma)
    
        return density


def build_t_grid(sigma, device, points_per_sigma: float = 5.0):
    target_dt = sigma / points_per_sigma
    width = 4000.0
    nt = int(width / target_dt) + 1
    nt = max(200, min(5000, nt))

    return torch.linspace(-2000, 2000, nt, device=device)


def estimate_density(model, m, k, sigma, loss):
    params = [p for p in model.parameters() if p.requires_grad]
    device = next(model.parameters()).device
    dtype = params[0].dtype

    n = sum(p.numel() for p in params)
    t_grid = build_t_grid(sigma, device)

    eigen = Eigenthings(
        model=model,
        m=m,
        k=k,
        t_grid=t_grid,
        sigma=sigma,
        params=params,
        device=device,
        loss=loss
        )

    density = torch.zeros_like(t_grid, dtype=dtype)

    for _ in range(k):
        v = torch.randn(n, device=device, dtype=dtype)
        v /= torch.linalg.norm(v)

        nodes, weights = eigen.lanczos_quadrature(v)
        density += eigen.density_from_lanczos(nodes, weights, t_grid, sigma)

    density /= k

    return density, t_grid