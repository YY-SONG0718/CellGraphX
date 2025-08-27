import optuna
import torch
import torch.optim as optim
from CellGraphX.models.model import HeteroGNN
from CellGraphX.training.trainer import Trainer
from CellGraphX.configs.config import Config
from CellGraphX.data.data_loader import data_loader, edge_weight_dict_loader
import logging
import sys


def objective(trial):

    config = Config()
    # params to optimise

    hidden_channels = trial.suggest_int("hidden_channels", 16, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    # initialize the model manually because we changed the hyper params
    model = HeteroGNN(
        hidden_channels=hidden_channels,
        out_channels=config.model.out_channels,
        num_layers=config.model.num_layers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Initialize the trainer with the model, data, optimizer, and config

    data = data_loader(config=config.data)
    edge_weight_dict = (
        edge_weight_dict_loader(data) if config.model.edge_weight else None
    )

    trainer = Trainer(
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        edge_weight_dict=edge_weight_dict,
        # optimizer, loss func, logdir build from config
    )

    return trainer.train(epochs=config.training.num_epochs)[0]  # return best val acc


def run_optuna(n_trials=50):

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"trial_optuna"  # Unique identifier of the study. Here we optimize for each test species
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction="maximize",  storage=storage_name, study_name=study_name, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)

    # Output the best hyperparameters found
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

    return study.best_trial
