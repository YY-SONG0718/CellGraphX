import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
import os

from collections import defaultdict
from torch_geometric.data import HeteroData
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing, GCNConv, HeteroConv, GraphConv, Linear
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt

import pickle

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


class SequenceEncoder(object):
    # The 'SequenceEncoder' encodes raw column strings into embeddings using a sentence transformer.
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(
            df.values,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
        )
        return x.cpu()


class GenresEncoder(object):
    # The 'GenreEncoder' splits the raw column strings by 'sep' and converts
    # individual elements to categorical labels.
    def __init__(self, sep="|"):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None, is_list=False):
        self.dtype = dtype
        self.is_list = is_list

    def __call__(self, df):
        if self.is_list:
            return torch.stack([torch.tensor(el) for el in df.values])
        return torch.from_numpy(df.values).to(self.dtype)
