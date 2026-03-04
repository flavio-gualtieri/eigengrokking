import torch
from typing import Tuple

def phi_mass_g1(phi: torch.Tensor, t: torch.Tensor, threshold: float = 1.0) -> float:
    # Ensure 1D
    phi = phi.reshape(-1)
    t = t.reshape(-1)

    # Total mass (should be ~1 if phi is normalized, but we normalize anyway)
    total_mass = torch.trapz(phi, t)

    # Mask region |t| > threshold and integrate only there
    mask = (t.abs() > threshold)
    if mask.sum() < 2:
        # Not enough points to integrate in that region
        return 0.0

    tail_mass = torch.trapz(phi[mask], t[mask])

    # Safe normalization
    frac = tail_mass / (total_mass + 1e-12)
    return float(frac.item())


def phi_mean_var(
        phi_avg: torch.Tensor,
        t: torch.Tensor,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if t.ndim != 1 or phi_avg.ndim != 1:
        raise ValueError("t and phi_avg must be 1D tensors")
    if t.shape != phi_avg.shape:
        raise ValueError("t and phi_avg must have the same shape")

    # Grid spacing (assumes roughly uniform spacing)
    dt = (t[1:] - t[:-1]).mean()

    # Total mass
    mass = (phi_avg.sum() * dt).clamp_min(eps)

    # Mean
    mu = (t * phi_avg).sum() * dt / mass

    # Variance
    var = (((t - mu) ** 2) * phi_avg).sum() * dt / mass

    return mu, var