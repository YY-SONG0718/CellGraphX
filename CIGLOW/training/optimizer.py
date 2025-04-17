import torch

torch.manual_seed(0)

import numpy as np

np.random.seed(0)

import random

random.seed(0)


def build_optimizer(model, config):
    if config.optimizer["name"] == "adam":
        return torch.optim.Adam(model.parameters(), **config.optimizer["params"])
