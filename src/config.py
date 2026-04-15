DATA_PATH = "data/processed/Dataset_Z.csv"
TARGET = "risk"

CATEGORICAL = ["state", "district", "season", "crop"]
NUMERICAL = ["year", "area", "production", "yield_log", "n", "p", "k", "ph", "soil_fertility"]
FEATURES = CATEGORICAL + NUMERICAL

MODEL_DIR = "models/tabnet"
MODEL_BASENAME = f"{MODEL_DIR}/crop_risk_tabnet"
MODEL_PATH = f"{MODEL_BASENAME}.zip"
PREPROCESSOR_PATH = f"{MODEL_DIR}/preprocessor.pkl"
METRICS_PATH = f"{MODEL_DIR}/metrics.json"

TABNET_PARAMS = {
    "n_d": 24,
    "n_a": 24,
    "n_steps": 4,
    "gamma": 1.3,
    "lambda_sparse": 1e-4,
}
