import torch


def nth_moment(density: torch.Tensor, t_grid: torch.Tensor, n: int) -> torch.Tensor:
    """
    Estimate the n-th moment:
        ∫ t^n rho(t) dt
    using the trapezoidal rule.
    """
    density = density.reshape(-1)
    t_grid = t_grid.reshape(-1)

    if density.shape != t_grid.shape:
        raise ValueError("density and t_grid must have the same shape")
    if n < 0:
        raise ValueError("n must be nonnegative")

    integrand = (t_grid ** n) * density
    return torch.trapz(integrand, t_grid)


def mass_above_thresh(density: torch.Tensor, t_grid: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """
    Estimate the mass above `threshold`:
        ∫_{t >= threshold} rho(t) dt

    For your case, 'mass over 1' is threshold=1.0.
    """
    density = density.reshape(-1)
    t_grid = t_grid.reshape(-1)

    if density.shape != t_grid.shape:
        raise ValueError("density and t_grid must have the same shape")

    mask = t_grid >= threshold
    if mask.sum() < 2:
        return torch.zeros((), dtype=density.dtype, device=density.device)

    return torch.trapz(density[mask], t_grid[mask])


def pos_neg_ratio(density: torch.Tensor, t_grid: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    density = density.reshape(-1)
    t_grid = t_grid.reshape(-1)

    if density.shape != t_grid.shape:
        raise ValueError("density and t_grid must have the same shape")

    pos_mask = t_grid > 0
    neg_mask = t_grid < 0

    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return torch.tensor(float("nan"), dtype=density.dtype, device=density.device)

    pos_mass = torch.trapz(density[pos_mask], t_grid[pos_mask])
    neg_mass = torch.trapz(density[neg_mask], t_grid[neg_mask])

    if torch.abs(neg_mass) < eps:
        if torch.abs(pos_mass) < eps:
            return torch.tensor(float("nan"), dtype=density.dtype, device=density.device)
        return torch.tensor(float("inf"), dtype=density.dtype, device=density.device)

    return pos_mass / neg_mass


def mirror_asymmetry(density: torch.Tensor, t_grid: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    density = density.reshape(-1)
    t_grid = t_grid.reshape(-1)

    if density.shape != t_grid.shape:
        raise ValueError("density and t_grid must have the same shape")

    # this assumes t_grid is symmetric and ordered ascending, e.g. linspace(-a, a, n)
    mirrored = torch.flip(density, dims=[0])

    numer = torch.trapz(torch.abs(density - mirrored), t_grid)
    denom = torch.trapz(torch.abs(density), t_grid)

    if torch.abs(denom) < eps:
        return torch.zeros((), dtype=density.dtype, device=density.device)

    return numer / (denom + eps)