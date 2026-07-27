# configs/registry.py

import torch
import torch.nn as nn


OPTIMIZERS = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}


LOSSES = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}