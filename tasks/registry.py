# tasks/registry.py

from __future__ import annotations

from tasks.mnist import MNIST_task
from tasks.mod_add import ModularAddition_task

TASKS = {
    "MNIST": MNIST_task,
    "MODULAR": ModularAddition_task,
}
