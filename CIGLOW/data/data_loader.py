import torch
import pandas as pd
import numpy as np
import pickle
import os

import warnings
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)


def read_pt(data_path):
    data = torch.load(data_path)
    return data


def split_train_val_test(data, split):

    # Now since i want to train on human and test on pt, i need to create the train val test split
    for name in ["train", "val", "test"]:
        idx = split[f"{name}_idx"]
        idx = torch.from_numpy(idx).to(torch.long)
        mask = torch.zeros(data["cell_type"].num_nodes, dtype=torch.bool)
        mask[idx] = True
        data["cell_type"][f"{name}_mask"] = mask
    return data


def get_split(species_val_now, species_test_now, species_origin):
    # using each of the other species as cross-validation
    print(f"now using {species_val_now} as validation set")

    df = pd.DataFrame(
        columns=[
            "case",
            "test_species",
            "val_species",
            "epoch",
            "train_acc",
            "train_loss",
            "val_acc",
            "test_acc",
        ]
    )

    split = {
        "train_idx": np.array(
            [
                k
                for k in species_origin.keys()
                if not (
                    species_origin[k] == species_val_now
                    or species_origin[k] == species_test_now
                )
            ]
        ),
        "val_idx": np.array(
            [k for k in species_origin.keys() if species_origin[k] == species_val_now]
        ),
        "test_idx": np.array(
            [k for k in species_origin.keys() if species_origin[k] == species_test_now]
        ),
    }

    return split


def load_species_origin_index_mappings(ct_index_mapping):

    with open(ct_index_mapping, "rb") as f:
        ct_sub_mapping = pickle.load(f)

    species_origin = dict(
        pd.Series(list(ct_sub_mapping.keys())).replace(".*_", "", regex=True)
    )

    return species_origin


def data_loader(config):
    # print(os.getcwd())
    data_orig = read_pt(
        data_path=f"./data/mtg_all_sp_wilcox_heterodata_only_gene_cell_edges.pt"
        # this will be run from cwd CIGLOW so get the paths right
    )
    species_origin = load_species_origin_index_mappings(
        "./data/mtg_all_sp_wilcox_data_ct_mapping.pkl"
    )
    split = {
        "train_idx": np.array(
            [
                k
                for k in species_origin.keys()
                if not (
                    species_origin[k] == config.val_species
                    or species_origin[k] == config.test_species
                )
            ]
        ),
        "val_idx": np.array(
            [
                k
                for k in species_origin.keys()
                if species_origin[k] == config.val_species
            ]
        ),
        "test_idx": np.array(
            [
                k
                for k in species_origin.keys()
                if species_origin[k] == config.test_species
            ]
        ),
    }

    data = split_train_val_test(data_orig, split)
    print(f"This is the HeteroData you are working with:\n{data}")
    return data
