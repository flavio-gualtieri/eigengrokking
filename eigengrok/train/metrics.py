import math
import torch
from eigengrok.registry import LOSSES


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

@torch.no_grad()
def compute_logit_stats(
    network,
    loader,
    device,
    N=None,
    saturation_tol: float = 1e-7,
    ):
    """
    (max |logit|, fraction of examples with a saturated softmax) over the
    first N examples of `loader`. Saturated: max softmax probability
    > 1 - saturation_tol. The softmax is computed in the logits' own dtype,
    as the cross-entropy loss is, and only the comparison is done in float64.
    In float32, a probability that close to 1 is stored as 1 or one rounding
    step below it, so the example's gradient is mostly lost to rounding
    (softmax collapse); in float64 it is not.
    """
    was_training = network.training
    network.eval()
    try:
        max_abs = 0.0
        saturated = 0
        seen = 0

        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            if not x.is_floating_point():
                x = x.float()

            logits = network(x)
            take = labels.size(0) if N is None else min(labels.size(0), N - seen)
            logits = logits[:take]

            max_abs = max(max_abs, logits.abs().max().item())
            p_max = torch.softmax(logits, dim=1).amax(dim=1)
            saturated += int((p_max.double() > 1 - saturation_tol).sum().item())
            seen += take

            if N is not None and seen >= N:
                break

        return max_abs, saturated / max(seen, 1)
    finally:
        network.train(was_training)
