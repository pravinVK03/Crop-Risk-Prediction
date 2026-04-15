from pathlib import Path

import pandas as pd
import torch

from .config import DEFAULT_CONFIG, ModelConfig, PipelineConfig
from .data import TabPreprocessor
from .model import TabTransformerClassifier


def _load_model_and_preprocessor(config: PipelineConfig = DEFAULT_CONFIG):
    checkpoint = torch.load(config.paths.checkpoint_path, map_location="cpu")
    preprocessor = TabPreprocessor.load(Path(config.paths.preprocessor_path))

    model_config = ModelConfig(**checkpoint["model_config"])
    model = TabTransformerClassifier(
        cat_cardinalities=checkpoint["cat_cardinalities"],
        num_features=checkpoint["num_features"],
        num_classes=checkpoint["num_classes"],
        config=model_config,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, preprocessor


def predict(frame: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    model, preprocessor = _load_model_and_preprocessor(config=config)
    cat_values, num_values = preprocessor.transform_features(frame)

    with torch.no_grad():
        logits = model(torch.from_numpy(cat_values), torch.from_numpy(num_values))
        probabilities = torch.softmax(logits, dim=1).numpy()
        pred_indices = probabilities.argmax(axis=1)

    pred_labels = [preprocessor.index_to_label[int(index)] for index in pred_indices]
    confidence = probabilities.max(axis=1)

    result = frame.copy()
    result["predicted_risk"] = pred_labels
    result["confidence"] = confidence
    return result
