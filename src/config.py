DATA_PATH = "data/processed/Dataset_Z.csv"
TARGET = "risk"

CATEGORICAL = ["state", "district", "season", "crop"]
NUMERICAL = ["year", "area", "production", "yield_log", "n", "p", "k", "ph", "soil_fertility"]
FEATURES = CATEGORICAL + NUMERICAL

MODEL_DIR = "models/tab_transformer"
MODEL_PATH = f"{MODEL_DIR}/crop_risk_tab_transformer.pt"
PREPROCESSOR_PATH = f"{MODEL_DIR}/preprocessor.pkl"
METRICS_PATH = f"{MODEL_DIR}/metrics.json"

TABTRANSFORMER_PARAMS = {
    "d_model": 64,
    "n_heads": 8,
    "n_layers": 3,
    "dropout": 0.1,
    "ff_multiplier": 4,
}

TRAIN_PARAMS = {
    "epochs": 10,
    "batch_size": 512,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "patience": 8,
    "grad_clip_norm": 1.0,
}
