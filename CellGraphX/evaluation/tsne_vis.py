import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import random

np.random.seed(0)

random.seed(0)


writer = SummaryWriter()

from sklearn.preprocessing import LabelEncoder


import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


import torch
import torch.optim as optim
from CellGraphX.training.trainer import Trainer
from CellGraphX.configs.config import Config
from CellGraphX.models.model import model_builder
from CellGraphX.data.data_loader import data_loader, edge_weight_dict_loader
from CellGraphX.training.optimizer import build_optimizer


class TSNE_vis:

    def __init__(self, model_path, data):
        self.model_path = model_path
        self.data = data
        self.species_origin = data.species_origin_index
        self.cell_type_index_mapping = data.cell_type_index_mapping

    def load_model(self):
        """Load trainned model from the given path"""

        config = Config()
        print("Initializing model...", flush=True)
        model = model_builder(config=config.model)
        print("Loading trainned model state dict...", flush=True)
        if osp.isfile(self.model_path):
            print(f"Loading model from {self.model_path}")
        else:
            raise FileNotFoundError(f"No model file found at {self.model_path}")

        model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
        )
        return model

    def get_tsne_embed(self, model):
        """Get model output embeddings after training and perform tSNE"""

        model.eval()

        edge_weight_dict = edge_weight_dict_loader(self.data)

        out = model(
            self.data.x_dict,
            self.data.edge_index_dict,
            edge_weight_dict=edge_weight_dict,
        )  # forward pass in eval mode

        print("Performing t-SNE ...", flush=True)

        tsne = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
        return tsne

    def plot_tsne(self, tsne, out_path):
        """Generate plot"""

        plt.figure(figsize=(9, 8))
        plt.xticks([])
        plt.yticks([])

        # prepare shapes

        print("Prepare species labels as shapes ...", flush=True)

        species_labels = self.species_origin.values()
        species_list = list(species_labels)

        label_encoder = LabelEncoder()
        species_integers = label_encoder.fit_transform(
            species_list
        )  # species list to integers for shape assignment
        shape = np.array(["o", "s", "^", "D", "v"])[
            species_integers
        ]  # assign dot shapes per species

        unique_shapes = np.unique(shape)

        # prepare color palette based on number of cell types

        print("Prepare cell type labels as colors ...", flush=True)
        color_labels = self.data["cell_type"].y
        color_labels = np.array(color_labels)
        num_colors = len(np.unique(color_labels))
        palette = sns.color_palette("hls", num_colors)
        color_map = {
            label: palette[i] for i, label in enumerate(np.unique(color_labels))
        }
        node_colors = np.array([color_map[label] for label in color_labels])

        # Marker legend mapping (species)
        marker_dicts = {
            "o": "C.jacchus",
            "s": "G.gorilla",
            "^": "H.sapiens",
            "D": "M.mulatta",
            "v": "P.troglodytes",
        }

        # Plot each species (marker)
        for marker in unique_shapes:
            idx = shape == marker
            plt.scatter(
                tsne[idx, 0],
                tsne[idx, 1],
                s=70,
                c=node_colors[idx],
                marker=marker,
                edgecolors="k",
                label=marker_dicts.get(marker, marker),
            )
        # Create a separate color legend for cell types
        species_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=marker_dicts.get(marker, marker),
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="k",
            )
            for marker in unique_shapes
        ]

        # cell type (color) legend
        cell_type_names = {v: k for k, v in self.cell_type_index_mapping.items()}
        color_handles = [
            Patch(
                facecolor=color,
                label=cell_type_names[label] if cell_type_names else f"Type {label}",
            )
            for label, color in color_map.items()
        ]

        # Combine the two legends

        print("plotting ...", flush=True)
        legend1 = plt.legend(
            handles=species_handles,
            title="Species (shape)",
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
        )
        plt.gca().add_artist(legend1)  # Keep the first legend when adding the second

        plt.legend(
            handles=color_handles,
            title="Cell Types (color)",
            loc="lower left",
            bbox_to_anchor=(1.05, 0),
        )

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

        plt.tight_layout()
        plt.savefig(
            f"{out_path}",  # note that this should be a complete image file path, with .png
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
