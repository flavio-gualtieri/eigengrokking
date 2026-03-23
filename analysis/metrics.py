import math
import torch
from configs.registry import LOSSES


@torch.no_grad()
def compute_accuracy(
    network,
    loader,
    device,
    N=None,
    dataset_name: str | None = None
    ):
    was_training = network.training
    network.eval()
    try:
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
            take = bs if N is None else min(bs, N - seen)

            correct += (pred[:take] == labels[:take]).sum()
            seen += take

            if N is not None and seen >= N:
                break

        return (correct.float() / max(seen, 1)).item()
    finally:
        network.train(was_training)


@torch.no_grad()
def compute_loss(
    network,
    loader,
    loss_function,
    device,
    N=None,
    dataset_name: str | None = None
    ):
    was_training = network.training
    network.eval()
    try:
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
            take = bs if N is None else min(bs, N - seen)

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
            if N is not None and seen >= N:
                break

        return (total / max(seen, 1)).item()
    finally:
        network.train(was_training)