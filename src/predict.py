from dataclasses import dataclass

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

from src.config import FEATURES, MODEL_PATH, PREPROCESSOR_PATH
from src.explain import explain_prediction, recommend_precautions
from src.preprocessing import TabularPreprocessor


@dataclass
class ModelBundle:
    model: TabNetClassifier
    preprocessor: TabularPreprocessor


def _risk_label_from_raw(raw_target) -> str:
    mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return mapping.get(int(raw_target), f"RISK_{raw_target}")


def load_model() -> ModelBundle:
    model = TabNetClassifier()
    model.load_model(MODEL_PATH)
    preprocessor = TabularPreprocessor.load(PREPROCESSOR_PATH)
    return ModelBundle(model=model, preprocessor=preprocessor)


def predict(bundle: ModelBundle, sample_df):
    encoded_x, _ = bundle.preprocessor.transform(sample_df, with_target=False)
    prediction_index = int(bundle.model.predict(encoded_x)[0])
    prediction_proba = bundle.model.predict_proba(encoded_x)[0]

    raw_target = bundle.preprocessor.index_to_target[prediction_index]
    risk_label = _risk_label_from_raw(raw_target)
    confidence = float(np.max(prediction_proba))
    class_probabilities = {}
    for class_index, raw_class in bundle.preprocessor.index_to_target.items():
        class_probabilities[_risk_label_from_raw(raw_class)] = float(prediction_proba[class_index])

    reasons = explain_prediction(
        model=bundle.model,
        preprocessor=bundle.preprocessor,
        encoded_x=encoded_x,
        raw_frame=sample_df,
        features=FEATURES,
        top_k=5,
    )
    precautions = recommend_precautions(risk_label=risk_label, reasons=reasons)

    return {
        "risk_class": int(raw_target),
        "risk_label": risk_label,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "explanation_method": "TabNet local feature attribution (per-sample), ranked by influence magnitude.",
        "explanation_note": "Higher contribution means stronger influence on this prediction; directionality is interpreted using value vs location reference.",
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
