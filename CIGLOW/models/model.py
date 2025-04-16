import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import HeteroConv, GCNConv, GraphConv


torch.manual_seed(0)
import random

random.seed(0)

writer = SummaryWriter()

import logging
import sys
import pickle
import pandas as pd
import numpy as np


np.random.seed(0)


def model_builder(config):
    if config.model.name == "graphconv":
        return HeteroGNN(**config.model.params)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("gene", "is_wilcox_marker_of", "cell_type"): GraphConv(
                        -1, hidden_channels
                    ),  # GraphConv can take edge weights
                    ("cell_type", "rev_is_wilcox_marker_of", "gene"): GraphConv(
                        -1, hidden_channels
                    ),
                    # Self-loop edges for genes and cell types
                    ("gene", "self_loop", "gene"): GraphConv(-1, hidden_channels),
                    ("cell_type", "self_loop", "cell_type"): GraphConv(
                        -1, hidden_channels
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        # edge_weight_dict is expected to be a dictionary with edge weights for each relation
        for conv in self.convs:
            # prepare the edge weight inputs for each relation type
            edge_weight_inputs = {}
            for relation, subconv in conv.convs.items():
                if (
                    isinstance(subconv, GraphConv)
                    and edge_weight_dict
                    and relation in edge_weight_dict
                ):
                    edge_weight_inputs[relation] = edge_weight_dict[relation]
                else:
                    edge_weight_inputs[relation] = None

            # apply the HeteroConv layer with edge weights if available
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_inputs)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict["cell_type"])
