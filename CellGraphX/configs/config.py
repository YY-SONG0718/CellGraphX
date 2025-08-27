class ModelConfig:
    model_name = "graphconv"
    hidden_channels = 64
    out_channels = 24  # equals to number of cell type classes
    num_layers = 2  # number of hops for two cell type nodes to reach each other
    edge_weight = True  # whether to use edge weights in GraphConv


class LossConfig:
    loss = {"name": "cross_entropy"}


class TrainingConfig:
    batch_size = 32
    num_epochs = 10
    optimizer = {"name": "adam", "params": {"lr": 1e-3, "weight_decay": 1e-5}}


class DataConfig:
    uri = "neo4j://localhost:7687"
    auth = ("test", "666666")
    # this output dir is only if the data is generated from the KG
    # which will be under the higher level data directory
    output_dir = "all_sp_heterodata_only_gene_cell_edges"

    edge_threshold = 3.0  # minimum avg_log2fc for wilcox marker genes to keep
    add_self_loops = False
    use_esm_embeddings = False  # example option

    heterodata_pt = "mtg_all_sp_wilcox_data_with_og_ct_name.pt"
    # currently with example data, it is placed under the same directory as data_loader.py
    # this path need to be changed if the data is generated from the KG
    # in this case, need to point the data loader to the heterodata_pt file in the higher level data directory and under output_dir
    species_origin_index = "mtg_all_sp_wilcox_data_with_og.pkl"  # this is the species origin of each cell type node
    val_species = "M.mulatta"
    test_species = "G.gorilla"

    cell_type_index_mapping = "mtg_all_sp_wilcox_data_ct_y_mapping.pkl"  # this is the mapping between cell type index (a number, essentially) and cell type name


class Config:
    model = ModelConfig
    loss = LossConfig
    training = TrainingConfig
    data = DataConfig
    log_dir = "../logs"  # TODO: this make the log dir at the same level to the dir whereever the config is initialised, yuyao see if this requires fixing
