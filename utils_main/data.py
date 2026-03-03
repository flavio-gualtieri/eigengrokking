from __future__ import annotations

from typing import Tuple, Optional
from pathlib import Path

import hashlib
import torch
import torchvision


def _modular_cache_path(download_directory: str, p: int) -> Path:
    cache_dir = Path(download_directory) / "MODULAR"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # include dtype in the key so float32/float64 caches don't collide
    dtype = str(torch.get_default_dtype()).replace("torch.", "")
    key = f"p={p}_dtype={dtype}"
    fname = hashlib.md5(key.encode("utf-8")).hexdigest() + ".pt"
    return cache_dir / fname


def _build_modular_tensors(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    a_vals = torch.arange(p)
    b_vals = torch.arange(p)
    A, B = torch.meshgrid(a_vals, b_vals, indexing="ij")

    a = A.reshape(-1)
    b = B.reshape(-1)

    inputs = torch.cat(
        [
            torch.nn.functional.one_hot(a, num_classes=p),
            torch.nn.functional.one_hot(b, num_classes=p),
        ],
        dim=1,
    ).to(torch.get_default_dtype())

    targets = ((a + b) % p).long()
    return inputs, targets


def get_torchvision_data(
        dataset: str = "MNIST",
        download_directory: str = ".",
        train_points: Optional[int] = 500,
        batch_size: int = 200,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        persistent_workers: bool = False,
        p: Optional[int] = None,  # Dummy arg for consistent signature with get_modular
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


def get_modular(
        download_directory: str = ".",
        train_points: Optional[float] = 0.8,
        batch_size: int = 200,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        persistent_workers: bool = False,
        p: int = 10,
        ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.DataLoader]:

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    cache_path = _modular_cache_path(download_directory, p)

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        inputs = payload["inputs"]
        targets = payload["targets"]
    else:
        inputs, targets = _build_modular_tensors(p)
        torch.save({"inputs": inputs.cpu(), "targets": targets.cpu()}, cache_path)

    full_dataset = torch.utils.data.TensorDataset(inputs, targets)
    total_points = len(full_dataset)

    if train_points is None:
        train = full_dataset
        test = full_dataset
    else:
        if not (0 < train_points <= 1):
            raise ValueError("train_points must be a float in (0, 1] representing a percentage.")

        n = int(train_points * total_points)
        n = max(1, min(n, total_points))

        train = torch.utils.data.Subset(full_dataset, range(n))
        test = torch.utils.data.Subset(full_dataset, range(n, total_points))

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