import torch

torch.manual_seed(0)

import numpy as np

np.random.seed(0)

import random

random.seed(0)


def get_loss_fn(config):
    if config.loss.name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {config.loss.name}")
