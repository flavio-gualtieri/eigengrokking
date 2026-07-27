# data/torchvision_data.py

from __future__ import annotations

from typing import Tuple, Optional
from pathlib import Path

import torch
import torchvision


def get_torchvision_data(
        dataset: str = "MNIST",
        download_directory: str = ".",
        train_points: Optional[int] = 500,
        batch_size: int = 200,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        persistent_workers: bool = False,
        ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Converts PIL image to [0,1] torch tensor
    transform = torchvision.transforms.ToTensor()

    # Load data
    if dataset == "MNIST":
        train_full = torchvision.datasets.MNIST(
            root=download_directory,
            train=True,
            transform=transform,
            download=True,
        )
        test = torchvision.datasets.MNIST(
            root=download_directory,
            train=False,
            transform=transform,
            download=True,
        )
    
    if dataset == "FashionMNIST":
        train_full = torchvision.datasets.FashionMNIST(
            root=download_directory,
            train=True,
            transform=transform,
            download=True,
        )
        test = torchvision.datasets.FashionMNIST(
            root=download_directory,
            train=False,
            transform=transform,
            download=True,
        )

    # Restrict to subset of training data if requested
    if train_points is None:
        train = train_full
    else:
        n = int(train_points)
        n = max(1, min(n, len(train_full)))
        train = torch.utils.data.Subset(train_full, range(n))

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=min(batch_size, len(train)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,
    )

    return train, test, train_loader