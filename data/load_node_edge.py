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

URI = "neo4j://localhost:7687"
AUTH = ("test", "666666")

with GraphDatabase.driver(URI, auth=AUTH) as driver:

    driver.verify_connectivity()


def fetch_data(query, parameters=None, **kwargs):
    with driver.session(database="mtg-wilcox") as session:
        result = session.run(query, parameters, **kwargs)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())


def load_node(
    cypher, index_col, encoders=None, category_col=None, parameters=None, **kwargs
):
    # Execute the cypher query and retrieve data from Neo4j
    df = fetch_data(cypher, parameters, **kwargs)
    df.set_index(index_col, inplace=True)
    # Define node mapping
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    # Define node features
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    y = None
    if category_col is not None:
        # Get unique categories and map to numerical labels
        categories = df[category_col].unique()
        category_to_idx = {cat: idx for idx, cat in enumerate(sorted(categories))}

        # Map category column to numerical labels
        y = df[category_col].map(category_to_idx).values
        y = torch.tensor(y, dtype=torch.long)  # length: n_nodes
        return x, y, mapping, category_to_idx
    return x, mapping


def load_edge(
    cypher,
    src_index_col,
    src_mapping,
    dst_index_col,
    dst_mapping,
    encoders=None,
    parameters=None,
    **kwargs
):
    # Execute the cypher query and retrieve data from Neo4j
    df = fetch_data(cypher, parameters, **kwargs)
    # Define edge index
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    # Define edge features
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr
