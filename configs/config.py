class ModelConfig:
    model_name = "graphconv"
    hidden_channels = 64
    out_channels = 21
    num_layers = 2


class LossConfig:
    loss = {"name": "cross_entropy"}


class TrainingConfig:
    batch_size = 32
    num_epochs = 100
    optimizer = {"name": "adam", "params": {"lr": 1e-3, "weight_decay": 1e-5}}


class DataConfig:
    uri = "neo4j://localhost:7687"
    auth = ("test", "666666")
    output_dir = "all_sp_heterodata_only_gene_cell_edges"
    edge_threshold = 3.0  # minimum avg_log2fc for wilcox marker genes to keep
    add_self_loops = True
    use_esm_embeddings = False  # example option
    val_species = "M.mulatta"
    test_species = "G.gorilla"


class Config:
    model = ModelConfig
    loss = LossConfig
    training = TrainingConfig
    data = DataConfig
    log_dir = "../logs"
