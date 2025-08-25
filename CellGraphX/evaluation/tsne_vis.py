
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import HeteroConv, GCNConv, GraphConv, GATv2Conv, DataParallel

torch.manual_seed(0)
import random

random.seed(0)


writer = SummaryWriter()

import logging
import sys
import pickle
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from training.trainer import Trainer
from configs.config import Config
from models.model import model_builder
from data.data_loader import data_loader
from training.optimizer import build_optimizer


np.random.seed(0)

config = Config()
print("Initializing model...", flush=True)
model = model_builder(config=config.model)

class TSNE_vis():

    def __init__(self, model_path):

    
    def load_model(self, model_path):

    def get_tsne_embed:
    
    def plot_tsne:
    
    