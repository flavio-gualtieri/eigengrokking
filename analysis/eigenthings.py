from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn


def get_params(model: nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def device() -> torch.device:
    if torch.cuda.is_available():
            return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")
    else:
        print("Using CPU device")
        return torch.device("cpu")


def pack_list(xs: Iterable[torch.Tensor]) -> torch.Tensor:
    xs = list(xs)
    if len(xs) == 0:
        return torch.empty(0)
    
    return torch.cat([x.reshape(-1) for x in xs], dim=0)


def unpack_vec(v: torch.Tensor, params: list[torch.Tensor]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    i = 0
    for p in params:
        n = p.numel()
        out.append(v[i : i + n].view_as(p))
        i += n

    return out


def default_loss_fn(
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_type: str = "cross_entropy",
    ) -> torch.Tensor:
    if loss_type == "cross_entropy":
        # logits: (B, C), y: (B,)
        return nn.functional.cross_entropy(logits, y)
    elif loss_type == "mse":
        # logits: (B, C), y: (B, C)
        return nn.functional.mse_loss(logits, y)
    else:
        raise ValueError(f"Unknown loss_type='{loss_type}'. Use 'cross_entropy' or 'mse'.")


def hvp(q):
    device = device()
    q = q.to(device)
    q_list = unpack_vec(q, params)

    logits = model(X)
    loss = loss_fn(logits, y)

    grads = torch.autograd.grad(loss, params, create_graph=True)

    gv = torch.zeros((), device=device, dtype=loss.dtype)
    for g, v in zip(grads, q_list):
        gv = gv + (g * v).sum()

    Hv = torch.autograd.grad(gv, params, retain_graph=False, create_graph=False)
    return pack_list(Hv)


def lanczos_T(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        m: int,
        *,
        v0: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        loss_type: str = "cross_entropy",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    # Obtain model parameters and device info
    device = model_device(model)
    params = get_params(model)
    n = sum(p.numel() for p in params)

    # Set up initial vector (v_0 may be provided, otherwise random)
    if v0 is None:
        v = torch.randn(n, device=device, dtype=dtype)
    else:
        v = v0.reshape(-1).to(device=device, dtype=dtype)

    # Normalize initial vector
    v = v / (v.norm() + 1e-12)

    # Initialize diagnoal and off-diagonal entries of T
    alphas = torch.zeros(m, device=device, dtype=dtype)
    betas = torch.zeros(max(m - 1, 0), device=device, dtype=dtype)

    # Previous vector and beta (init to zero for first iteration)
    v_prev = torch.zeros_like(v)
    beta_prev = torch.zeros((), device=device, dtype=dtype)

    for j in range(m):
        # Compute w = H v_j
        w = hvp(model, X, y, v, loss_fn=loss_fn)

        # Subtract previous direction
        if j > 0:
            w = w - beta_prev * v_prev

        # Compute alpha
        alpha = (v * w).sum()

        # Subtract alpha * v from w to orthogonalize against current vector
        w = w - alpha * v

        # Compute beta for next iteration (except on last iteration)
        beta = w.norm()

        alphas[j] = alpha
        if j < m - 1:
            betas[j] = beta

        v_prev = v
        beta_prev = beta

        # Update v for next iteration (except on last iteration)
        if j < m - 1:
            v = w / (beta + 1e-20)

    # Construct T from alphas and betas
    T = torch.diag(alphas)
    if m > 1:
        T = T + torch.diag(betas, 1) + torch.diag(betas, -1)

    return T


def lanczos_nodes_weights(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        m: int,
        *,
        v0: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        loss_type: str = "cross_entropy",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # Obtain tridiagonal matrix T
    T = lanczos_T(model, X, y, m, v0=v0, loss_fn=loss_fn, loss_type=loss_type)

    # Compute on CPU
    T_cpu = T.detach().to("cpu")

    # Diagonalize T to get nodes (eigenvalues) and weights
    l_cpu, U_cpu = torch.linalg.eigh(T_cpu)
    l = l_cpu.to(T.device)
    U = U_cpu.to(T.device)

    # Weights are square of first component
    w = U[0, :].pow(2)
    
    return l, w


# COMBINE GAUSSIAN KERNEL AND PHI_FROM_LW??

def gaussian_kernel(l: torch.Tensor, t: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_t = torch.as_tensor(sigma, device=t.device, dtype=t.dtype)
    z = (t[None, :] - l[:, None]) / (sigma_t + 1e-20)
    return torch.exp(-0.5 * z * z) / (sigma_t * math.sqrt(2.0 * math.pi))


def phi_from_lw(l: torch.Tensor, w: torch.Tensor, t: torch.Tensor, sigma: float) -> torch.Tensor:
    K = gaussian_kernel(l, t, sigma)  # (m, nt)
    return (w[:, None] * K).sum(dim=0)  # (nt,)


def estimate_hessian_extremes(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        m_pilot: int,
        k_pilot: int,
        *,
        generator: Optional[torch.Generator] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        loss_type: str = "cross_entropy",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    params = get_params(model)
    n = sum(p.numel() for p in params)
    device = model_device(model)

    lambda_min_est: Optional[torch.Tensor] = None
    lambda_max_est: Optional[torch.Tensor] = None

    for _ in range(k_pilot):
        v0 = torch.randn(n, device=device, dtype=dtype, generator=generator)
        l, _w = lanczos_nodes_weights(
            model, X, y, m_pilot, v0=v0, loss_fn=loss_fn, loss_type=loss_type
        )

        l_min = l.min()
        l_max = l.max()

        lambda_min_est = l_min if lambda_min_est is None else torch.minimum(lambda_min_est, l_min)
        lambda_max_est = l_max if lambda_max_est is None else torch.maximum(lambda_max_est, l_max)

    # Both must exist since k_pilot>=1
    return lambda_min_est, lambda_max_est


def choose_t_from_extremes(
        lambda_min: torch.Tensor,
        lambda_max: torch.Tensor,
        sigma: float,
        *,
        c_margin: float = 4.0,     # margin = c_margin * sigma
        inflate: float = 1.10,     # widen interval a bit to be safe
        max_points: int = 5000,
        min_points: int = 200,
        points_per_sigma: float = 5.0,  # dt ≈ sigma / points_per_sigma
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    device = lambda_min.device
    sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype)

    margin = c_margin * sigma_t

    # Inflate outward (careful with sign)
    t_min = (lambda_min - margin)
    t_max = (lambda_max + margin)

    # Expand interval about center by `inflate`
    center = 0.5 * (t_min + t_max)
    half_width = 0.5 * (t_max - t_min) * inflate
    t_min = center - half_width
    t_max = center + half_width

    # Choose nt so dt ≲ sigma/points_per_sigma
    target_dt = sigma_t / points_per_sigma
    width = (t_max - t_min).clamp_min(target_dt)
    nt = int(torch.ceil(width / target_dt).item()) + 1
    nt = max(min_points, min(max_points, nt))

    return torch.linspace(t_min.item(), t_max.item(), nt, device=device, dtype=dtype)


def lanczos_phi_average(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        m: int, # Size of Kyrlov subspace
        k: int, # Number of probes
        sigma: float,
        *,
        generator: Optional[torch.Generator] = None, # Maybe remove
        eps: float = 1e-8,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        loss_type: str = "cross_entropy",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if k < 1:
        raise ValueError("k must be >= 1")

    # Obtain model parameters and device info
    params = get_params(model)
    n = sum(p.numel() for p in params)
    device = model_device(model)

    lam_min, lam_max = estimate_hessian_extremes(
        model,
        X,
        y,
        m_pilot=30,
        k_pilot=10,
        generator=generator,
        loss_fn=loss_fn,
        loss_type=loss_type,
        dtype=dtype
    )

    t = choose_t_from_extremes(
        lam_min,
        lam_max,
        sigma,
        c_margin=4.0,
        inflate=1.10,
        points_per_sigma=5.0,
        min_points=200,
        max_points=5000,
        dtype=dtype
    )

    # Initialize phi sum and spectral bounds
    phi_sum = torch.zeros_like(t)
    lambda_max_est: Optional[torch.Tensor] = None
    lambda_min_pos_est: Optional[torch.Tensor] = None

    # Lanczos iterations with different random probes
    for _ in range(k):
        v0 = torch.randn(n, device=device, dtype=dtype, generator=generator)

        # Compute Lanczos nodes and weights for probe
        l, w = lanczos_nodes_weights(model, X, y, m, v0=v0, loss_fn=loss_fn, loss_type=loss_type)

        # Update phi sum with contribution from probe
        phi_sum = phi_sum + phi_from_lw(l, w, t, sigma)

        l_max = l.max()
        pos = l[l > eps]
        l_min_pos = pos.min() if pos.numel() > 0 else None
        

        lambda_max_est = l_max if lambda_max_est is None else torch.maximum(lambda_max_est, l_max)
        lambda_min_pos_est = l_min_pos if lambda_min_pos_est is None else (
            l_min_pos if lambda_min_pos_est is None else torch.minimum(lambda_min_pos_est, l_min_pos)
        )

    phi_avg = phi_sum / float(k)

    if lambda_max_est is None or lambda_min_pos_est is None or (lambda_min_pos_est.abs() < eps):
        kappa = torch.as_tensor(float("inf"), device=device, dtype=dtype)
    else:
        kappa = lambda_max_est / lambda_min_pos_est

    return phi_avg, kappa, t