# config.py
class Config:
    # Model-specific settings
    model_name = "graphconv"
    loss = {"name": "cross_entropy"}
    optimizer = {"name": "adam", "params": {"lr": 1e-3, "weight_decay": 1e-5}}
    # model params
    hidden_channels = 64
    out_channels = 21
    num_layers = 2

    # Training settings
    batch_size = 32
    num_epochs = 100

    # Logging settings
    log_dir = "../logs"

    # define train test val species
    val_species = "P.troglodytes"
    test_species = "M.mulatta"
