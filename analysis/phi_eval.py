import torch
from typing import Tuple

def phi_mean_var(phi, t):
    mass = torch.trapz(phi, t)
    mu = torch.trapz(t * phi, t) / mass
    var = torch.trapz((t - mu)**2 * phi, t) / mass

    return mu, var


def phi_moment(
        phi: torch.Tensor,
        t: torch.Tensor,
        order: int,
        central: bool = False,
        eps: float = 1e-12,
    ) -> torch.Tensor:
    phi = phi.reshape(-1)
    t = t.reshape(-1)

    # Total mass
    mass = torch.trapz(phi, t).clamp_min(eps)

    if central:
        mu = torch.trapz(t * phi, t) / mass
        values = (t - mu) ** order
    else:
        values = t ** order

    moment = torch.trapz(values * phi, t) / mass
    return moment


def phi_tail_mass(phi, t, threshold=1.0):
    mask = (t.abs() > threshold).float()
    tail = torch.trapz(phi * mask, t)
    total = torch.trapz(phi, t)

    return tail / total


def phi_pos_neg_ratio(phi: torch.Tensor, t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    phi = phi.reshape(-1)
    t = t.reshape(-1)

    pos_mask = (t > 0).float()
    neg_mask = (t < 0).float()

    pos_mass = torch.trapz(phi * pos_mask, t)
    neg_mass = torch.trapz(phi * neg_mask, t)

    print(f"Positive mass: {pos_mass:.4e}, Negative mass: {neg_mass:.4e}")

    return pos_mass / (neg_mass + eps)