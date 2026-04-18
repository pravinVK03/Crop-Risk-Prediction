from dataclasses import dataclass

import numpy as np
import torch

from src.config import CATEGORICAL, FEATURES, MODEL_PATH, PREPROCESSOR_PATH
from src.explain import explain_prediction, recommend_precautions
from src.preprocessing import TabularPreprocessor
from src.tab_transformer.config import ModelConfig
from src.tab_transformer.model import TabTransformerClassifier


@dataclass
class ModelBundle:
    model: TabTransformerClassifier
    preprocessor: TabularPreprocessor
    checkpoint_meta: dict


def _risk_label_from_raw(raw_target) -> str:
    mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return mapping.get(int(raw_target), f"RISK_{raw_target}")


def load_model() -> ModelBundle:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model_cfg = ModelConfig(**checkpoint["model_config"])
    model = TabTransformerClassifier(
        cat_cardinalities=checkpoint["cat_cardinalities"],
        num_features=checkpoint["num_features"],
        num_classes=checkpoint["num_classes"],
        config=model_cfg,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = TabularPreprocessor.load(PREPROCESSOR_PATH)
    return ModelBundle(model=model, preprocessor=preprocessor, checkpoint_meta=checkpoint)


def predict_batch(bundle: ModelBundle, frame):
    x, _ = bundle.preprocessor.transform(frame, with_target=False)
    x_cat = torch.from_numpy(x[:, : len(CATEGORICAL)].astype(np.int64))
    x_num = torch.from_numpy(x[:, len(CATEGORICAL) :].astype(np.float32))

    with torch.no_grad():
        logits = bundle.model(x_cat, x_num)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        pred_indices = probabilities.argmax(axis=1)
    return pred_indices, probabilities


def predict(bundle: ModelBundle, sample_df):
    pred_indices, probabilities = predict_batch(bundle, sample_df)
    prediction_index = int(pred_indices[0])
    prediction_proba = probabilities[0]

    raw_target = bundle.preprocessor.index_to_target[prediction_index]
    risk_label = _risk_label_from_raw(raw_target)
    confidence = float(np.max(prediction_proba))
    class_probabilities = {}
    for class_index, raw_class in bundle.preprocessor.index_to_target.items():
        class_probabilities[_risk_label_from_raw(raw_class)] = float(prediction_proba[class_index])

    reasons = explain_prediction(
        bundle=bundle,
        raw_frame=sample_df,
        features=FEATURES,
        pred_index=prediction_index,
        top_k=5,
    )
    precautions = recommend_precautions(risk_label=risk_label, reasons=reasons)

    return {
        "risk_class": int(raw_target),
        "risk_label": risk_label,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "explanation_method": "TabTransformer perturbation-based local attribution (feature replacement against location/historical references).",
        "explanation_note": "Contribution is measured by how much predicted-class probability changes when one feature is replaced with its reference value.",
        "reasons": reasons,
        "precautions": precautions,
    }


def predict_from_api_payload(payload: dict, bundle: ModelBundle | None = None):
    active_bundle = bundle if bundle is not None else load_model()
    input_frame = active_bundle.preprocessor.api_payload_to_frame(payload)
    result = predict(active_bundle, input_frame)
    location = payload.get("location", {})
    soil = payload.get("soil", {})
    inferred_fields = []

    if "season" not in payload:
        inferred_fields.append("season")
    for field in ["n", "p", "k", "ph", "soil_fertility"]:
        has_direct = field in payload
        has_nested = field in soil
        if not has_direct and not has_nested:
            inferred_fields.append(field)
    for field in ["year", "area", "production", "yield_log"]:
        if field not in payload:
            inferred_fields.append(field)
    if "state" not in payload and "state" not in location:
        inferred_fields.append("state")
    if "district" not in payload and "district" not in location:
        inferred_fields.append("district")

    return {
        "input": input_frame.iloc[0].to_dict(),
        "inferred_fields": inferred_fields,
        "prediction": result,
    }
