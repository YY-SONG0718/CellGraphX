# main.py

import torch
import torch.optim as optim
from training.trainer import Trainer
from configs.config import Config
from models.model import model_builder
from data.data_loader import data_loader, edge_weight_dict_loader
from training.optimizer import build_optimizer
from evaluation.tsne_vis import TSNE_vis


def main():
    # Load configuration
    print("Loading configuration...", flush=True)

    config = Config()
    # Initialize the model
    print("Initializing model...", flush=True)
    model = model_builder(config=config.model)

    # Prepare the dataset and dataloaders
    print("Loading data...", flush=True)
    data = data_loader(config=config.data)
    edge_weight_dict = (
        edge_weight_dict_loader(data) if config.model.edge_weight else None
    )

    # Optimizer and learning rate scheduler
    optimizer = build_optimizer(model, config.training)

    # Scheduler (optional, for learning rate decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Initialize the trainer with the model, data, optimizer, and config

    print("Initializing trainer...", flush=True)
    trainer = Trainer(
        model=model,
        data=data,
        scheduler=scheduler,
        config=config,
        edge_weight_dict=edge_weight_dict,
        # optimizer, loss func, logdir build from config
    )

    # Start the training
    print("Starting training...", flush=True)
    trainer.train(epochs=config.training.num_epochs)

    print("Training completed with model saved.", flush=True)

    # t-SNE visualization
    vis = TSNE_vis(model_path=f"{trainer.log_dir}/best.pt", data=data)

    vis.plot_tsne(
        vis.get_tsne_embed(vis.load_model()),
        out_path=f"{trainer.log_dir}/tsne_plot.png",
    )


if __name__ == "__main__":
    main()
