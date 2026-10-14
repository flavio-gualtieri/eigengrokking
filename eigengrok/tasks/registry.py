# eigengrok/tasks/registry.py

from __future__ import annotations

from eigengrok.tasks.mnist import MNIST_task
from eigengrok.tasks.modular import ModularAddition_task

TASKS = {
    "MNIST": MNIST_task,
    "FashionMNIST": MNIST_task,  # same MLP/pipeline; build_data passes cfg.task through as the dataset name
    "MODULAR": ModularAddition_task,
}


def get_task(cfg):
    """Instantiates the Task named by cfg.task."""
    if cfg.task not in TASKS:
        raise ValueError(f"Unknown task {cfg.task!r}. Options: {list(TASKS)}")
    return TASKS[cfg.task](cfg)
