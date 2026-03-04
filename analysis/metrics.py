import math
import torch
from configs.registry import LOSSES

@torch.no_grad()
def compute_accuracy(network, dataset, device, N=2000, batch_size=2048, dataset_name: str | None = None,
                     num_workers: int = 4):
    was_training = network.training
    network.eval()
    try:
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        correct = torch.zeros((), device=device, dtype=torch.long)
        seen = 0

        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if dataset_name == "MNIST" and x.ndim > 2:
                x = x.view(x.size(0), -1).float()
            elif not x.is_floating_point():
                x = x.float()

            logits = network(x)
            pred = logits.argmax(dim=1)

            bs = labels.size(0)
            take = min(bs, N - seen)

            correct += (pred[:take] == labels[:take]).sum()
            seen += take
            if seen >= N:
                break

        return (correct.float() / max(seen, 1)).item()
    finally:
        network.train(was_training)


@torch.no_grad()
def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=2048, dataset_name: str | None = None,
                 num_workers: int = 4):
    was_training = network.training
    network.eval()
    try:
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        loss_fn = LOSSES[loss_function](reduction="sum")

        total = torch.zeros((), device=device, dtype=torch.float32)
        seen = 0
        one_hots = None

        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if dataset_name == "MNIST" and x.ndim > 2:
                x = x.view(x.size(0), -1).float()
            elif not x.is_floating_point():
                x = x.float()

            y = network(x)

            bs = labels.size(0)
            take = min(bs, N - seen)

            if loss_function == "CrossEntropy":
                total += loss_fn(y[:take], labels[:take])
            elif loss_function == "MSE":
                C = y.size(1)
                if one_hots is None or one_hots.size(0) != C:
                    one_hots = torch.eye(C, device=device, dtype=y.dtype)
                total += loss_fn(y[:take], one_hots[labels[:take]])
            else:
                raise ValueError(f"Unknown loss_function: {loss_function}")

            seen += take
            if seen >= N:
                break

        return (total / max(seen, 1)).item()
    finally:
        network.train(was_training)