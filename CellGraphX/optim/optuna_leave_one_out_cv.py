import optuna
import torch
import torch.optim as optim
from CellGraphX.models.model import HeteroGNN
from CellGraphX.training.trainer import Trainer
from CellGraphX.configs.config import Config
from CellGraphX.data.data_loader import (
    data_loader_no_split,
    edge_weight_dict_loader,
    split_train_val_test,
)
import logging
import sys
import numpy as np


def objective(trial):

    config = Config()
    # params to optimise

    hidden_channels = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # initialize the model manually because we changed the hyper params

    print("Initializing model...", flush=True)
    model = HeteroGNN(
        hidden_channels=hidden_channels,
        out_channels=config.model.out_channels,
        num_layers=config.model.num_layers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Initialize the trainer with the model, data, optimizer, and config
    print("Loading data...", flush=True)

    data_no_split = data_loader_no_split(config=config.data)

    edge_weight_dict = (
        edge_weight_dict_loader(data_no_split) if config.model.edge_weight else None
    )

    species_origin = data_no_split.species_origin_index

    val_scores = []

    for val_species in config.data.all_species:

        print(f"Leave one out CV, val species at the moment is: {val_species}", flush=True)

        split = {
            "train_idx": np.array(
                [
                    k
                    for k in species_origin.keys()
                    if not (species_origin[k] == val_species)
                ]
            ),
            "val_idx": np.array(
                [k for k in species_origin.keys() if species_origin[k] == val_species]
            ),
        }
        print(f"Split train val and test species", flush=True)

        data = split_train_val_test(data_no_split, split)

        print("Initializing trainer...", flush=True)

        trainer = Trainer(
            model=model,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            edge_weight_dict=edge_weight_dict,
            log_dir=f"./logs/optuna_trial_{trial.number}",
            # optimizer, loss func, logdir build from config
        )

        print(
            f"hidden_channels: {hidden_channels}, dropout: {dropout}, lr: {lr}, weight_decay: {weight_decay}",
            flush=True,
        )

        val_acc = trainer.train(epochs=config.training.num_epochs)[0]

        print(f"Val acc for species {val_species}: {val_acc}", flush=True)      

    val_scores.append(val_acc)
    # return simple mean of each val species acc

    return float(np.mean(val_scores))


def run_optuna(n_trials=50):
    print(
        "Starting hyperparameter optimization with Optuna, leave-one-out cross-validation...",
        flush=True,
    )
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"trial_optuna_cv"  # Unique identifier of the study. Here we optimize for each test species
    storage_name = "sqlite:///{}.db".format(study_name)

    print("Running Optuna study...", flush=True)
    study = optuna.create_study(
        direction="maximize",
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    # Output the best hyperparameters found
    print(f"Best trial: {study.best_trial.value}", flush=True)
    print(f"Best params: {study.best_trial.params}", flush=True)

    return study.best_trial
