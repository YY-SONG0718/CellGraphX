import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from training.losses import get_loss_fn
from training.optimizer import build_optimizer
import logging


class Trainer:
    def __init__(
        self,
        model,
        data,
        config,
        scheduler=None,
        device="cpu",
        edge_weight_dict=None,
        log_dir="./logs",
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.optimizer = build_optimizer(model, config.training)
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_weight_dict = edge_weight_dict
        self.config = config
        self.loss_fn = get_loss_fn(config.loss)
        self.log_dir = config.log_dir

        # Logging setup
        os.makedirs(log_dir, exist_ok=True)
        self._setup_logger(log_dir)
        self.logger = logging.getLogger(__name__)

        self.best_val_acc = 0.0
        self.log_dir = log_dir

    def _setup_logger(self, log_dir):
        log_file = os.path.join(log_dir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def train(self, epochs):
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            train_loss = self._train_one_epoch()
            accs = self._evaluate()
            train_acc, val_acc, test_acc = accs

            self.logger.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, is_best=True)
            self._save_checkpoint(epoch, is_best=False)

    def _train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        mask = self.data["cell_type"].train_mask
        loss = self.loss_fn(
            out[mask], self.data["cell_type"].y[mask]
        )  # only use training data for back prop

        loss.backward()
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return float(loss)  # training loss

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        pred = out.argmax(dim=-1)

        accs = []
        for split in ["train_mask", "val_mask", "test_mask"]:
            mask = self.data["cell_type"][split]
            acc = (pred[mask] == self.data["cell_type"].y[mask]).sum() / mask.sum()
            accs.append(float(acc))

        return accs

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        filename = os.path.join(self.log_dir, "best.pt" if is_best else "latest.pt")
        torch.save(checkpoint, filename)
        self.logger.info(f"Checkpoint saved to {filename}")
