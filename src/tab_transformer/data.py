import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from .config import DataConfig


@dataclass
class SplitData:
    train: tuple[np.ndarray, np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray, np.ndarray]


class TabPreprocessor:
    def __init__(self, categorical_columns: tuple[str, ...], numerical_columns: tuple[str, ...], target: str):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target = target
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numerical_mean: np.ndarray | None = None
        self.numerical_std: np.ndarray | None = None
        self.label_to_index: dict[int | str, int] = {}
        self.index_to_label: dict[int, int | str] = {}

    def fit(self, frame: pd.DataFrame) -> None:
        for column in self.categorical_columns:
            categories = sorted(frame[column].astype(str).unique().tolist())
            # Index 0 is reserved for unknown values.
            self.category_maps[column] = {value: index + 1 for index, value in enumerate(categories)}

        num_values = frame.loc[:, self.numerical_columns].astype(float).values
        self.numerical_mean = num_values.mean(axis=0)
        self.numerical_std = num_values.std(axis=0)
        self.numerical_std[self.numerical_std == 0] = 1.0

        labels = sorted(frame[self.target].unique().tolist())
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.index_to_label = {index: label for label, index in self.label_to_index.items()}

    def transform_features(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cat_matrix = np.zeros((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for column_index, column in enumerate(self.categorical_columns):
            mapping = self.category_maps[column]
            cat_matrix[:, column_index] = frame[column].astype(str).map(mapping).fillna(0).astype(np.int64).values

        num_matrix = frame.loc[:, self.numerical_columns].astype(float).values.astype(np.float32)
        num_matrix = ((num_matrix - self.numerical_mean) / self.numerical_std).astype(np.float32)
        return cat_matrix, num_matrix

    def transform_target(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.target].map(self.label_to_index).astype(np.int64).values

    def fit_transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fit(frame)
        cat_matrix, num_matrix = self.transform_features(frame)
        targets = self.transform_target(frame)
        return cat_matrix, num_matrix, targets

    def to_torch_dataset(self, cat_matrix: np.ndarray, num_matrix: np.ndarray, targets: np.ndarray) -> TensorDataset:
        cat_matrix = np.ascontiguousarray(cat_matrix)
        num_matrix = np.ascontiguousarray(num_matrix)
        targets = np.ascontiguousarray(targets)
        return TensorDataset(
            torch.from_numpy(cat_matrix),
            torch.from_numpy(num_matrix),
            torch.from_numpy(targets),
        )

    @property
    def categorical_cardinalities(self) -> list[int]:
        return [len(self.category_maps[column]) + 1 for column in self.categorical_columns]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "TabPreprocessor":
        with path.open("rb") as handle:
            return pickle.load(handle)


def load_frame(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def build_splits(frame: pd.DataFrame, preprocessor: TabPreprocessor, config: DataConfig) -> SplitData:
    train_df, test_df = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=frame[config.target],
    )

    val_fraction = config.validation_size / (1.0 - config.test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_fraction,
        random_state=config.random_state,
        stratify=train_df[config.target],
    )

    preprocessor.fit(train_df)

    train_split = (*preprocessor.transform_features(train_df), preprocessor.transform_target(train_df))
    val_split = (*preprocessor.transform_features(val_df), preprocessor.transform_target(val_df))
    test_split = (*preprocessor.transform_features(test_df), preprocessor.transform_target(test_df))
    return SplitData(train=train_split, val=val_split, test=test_split)
