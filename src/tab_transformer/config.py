from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    data_path: str = "data/processed/Dataset_Z.csv"
    target: str = "risk"
    categorical_columns: tuple[str, ...] = ("state", "district", "season", "crop")
    numerical_columns: tuple[str, ...] = (
        "year",
        "area",
        "production",
        "yield_log",
        "n",
        "p",
        "k",
        "ph",
        "soil_fertility",
    )
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.1
    ff_multiplier: int = 4


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 512
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 5


@dataclass(frozen=True)
class PathConfig:
    artifact_dir: str = "models/tab_transformer"
    checkpoint_name: str = "model.pt"
    preprocessor_name: str = "preprocessor.pkl"
    metrics_name: str = "metrics.json"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.artifact_dir) / self.checkpoint_name

    @property
    def preprocessor_path(self) -> Path:
        return Path(self.artifact_dir) / self.preprocessor_name

    @property
    def metrics_path(self) -> Path:
        return Path(self.artifact_dir) / self.metrics_name


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    paths: PathConfig = PathConfig()


DEFAULT_CONFIG = PipelineConfig()
