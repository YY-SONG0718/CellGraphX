import os
import sys

import torch
import torch.optim as optim
import os.path as osp

sys.path.append(os.path.abspath(".."))
from CellGraphX.optim.optuna import run_optuna

best_trial = run_optuna(n_trials=10)
print("Best hyperparameters:", best_trial.params)

