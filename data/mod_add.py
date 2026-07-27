# data/mod_add.py

from __future__ import annotations

from typing import Optional, Tuple

import torch

# Token id for "=" is `modulus` itself, one past the largest number token.
N_CTX = 3  # [a, b, =]


def get_modular_addition_data(
        modulus: int = 97,
        train_frac: float = 0.3,
        batch_size: int = 512,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    """
    Builds the "a + b = ?" (mod `modulus`) dataset used in grokking studies.
    Enumerates every (a, b) pair as the token sequence [a, b, EQUALS], labelled
    with (a + b) % modulus, then splits into train/test by `train_frac`.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    equals_token = modulus

    a = torch.arange(modulus).repeat_interleave(modulus)
    b = torch.arange(modulus).repeat(modulus)
    equals = torch.full_like(a, equals_token)

    inputs = torch.stack([a, b, equals], dim=1).long()
    labels = ((a + b) % modulus).long()

    n = inputs.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))

    n_train = int(round(train_frac * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train = torch.utils.data.TensorDataset(inputs[train_idx], labels[train_idx])
    test = torch.utils.data.TensorDataset(inputs[test_idx], labels[test_idx])

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=min(batch_size, len(train)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train, test, train_loader
