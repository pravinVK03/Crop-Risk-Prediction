import numpy as np
import torch

from src.config import CATEGORICAL, NUMERICAL


FEATURE_LABELS = {
    "state": "State",
    "district": "District",
    "season": "Season",
    "crop": "Crop",
    "year": "Year",
    "area": "Area",
    "production": "Production",
    "yield_log": "Yield (log)",
    "n": "Nitrogen (N)",
    "p": "Phosphorus (P)",
    "k": "Potassium (K)",
    "ph": "Soil pH",
    "soil_fertility": "Soil Fertility",
}


def _predict_prob(bundle, frame):
    x, _ = bundle.preprocessor.transform(frame, with_target=False)
    x_cat = torch.from_numpy(x[:, : len(CATEGORICAL)].astype(np.int64))
    x_num = torch.from_numpy(x[:, len(CATEGORICAL) :].astype(np.float32))
    with torch.no_grad():
        logits = bundle.model(x_cat, x_num)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return prob


def _reference_value(preprocessor, raw_frame, feature):
    state = str(raw_frame.iloc[0]["state"]).strip().lower()
    district = str(raw_frame.iloc[0]["district"]).strip().lower()
    crop = str(raw_frame.iloc[0]["crop"]).strip().lower()

    if feature in NUMERICAL:
        profile = preprocessor._fallback_profile(state=state, district=district)
        return float(profile.get(feature, preprocessor.numeric_medians.get(feature, 0.0))), "location_profile"
    if feature == "season":
        return preprocessor.infer_season(state=state, district=district, crop=crop), "historical_mode"
    if feature in CATEGORICAL:
        return preprocessor.category_modes.get(feature, "unknown"), "global_mode"
    return None, "not_available"


def _format_value(feature, value):
    if feature in NUMERICAL:
        return float(value)
    return str(value)


def explain_prediction(bundle, raw_frame, features, pred_index: int, top_k: int = 5):
    base_prob = _predict_prob(bundle, raw_frame)
    base_target_prob = float(base_prob[pred_index])

    influences = []
    for feature in features:
        perturbed = raw_frame.copy()
        ref_value, ref_source = _reference_value(bundle.preprocessor, raw_frame, feature)
        if ref_value is None:
            continue

        perturbed.loc[:, feature] = ref_value
        pert_prob = _predict_prob(bundle, perturbed)
        pert_target_prob = float(pert_prob[pred_index])

        support_delta = base_target_prob - pert_target_prob
        influence_score = abs(support_delta)
        original_val = raw_frame.iloc[0][feature]

        if support_delta > 0:
            direction = "supports"
        elif support_delta < 0:
            direction = "reduces"
        else:
            direction = "neutral"

        if feature in NUMERICAL:
            deviation = float(original_val) - float(ref_value)
            reason = (
                f"{FEATURE_LABELS.get(feature, feature)} {direction} this prediction; "
                f"value is {deviation:+.3f} vs reference."
            )
        else:
            match_text = "matches" if str(original_val).lower() == str(ref_value).lower() else "differs from"
            reason = (
                f"{FEATURE_LABELS.get(feature, feature)} {direction} this prediction; "
                f"input {match_text} reference pattern."
            )

        influences.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "value": _format_value(feature, original_val),
                "reference_value": _format_value(feature, ref_value),
                "reference_source": ref_source,
                "contribution": float(support_delta),
                "abs_contribution": float(influence_score),
                "direction": direction,
                "reason": reason,
            }
        )

    total_abs = sum(item["abs_contribution"] for item in influences) or 1.0
    for item in influences:
        item["contribution_pct"] = 100.0 * item["abs_contribution"] / total_abs

    ranked = sorted(influences, key=lambda row: row["abs_contribution"], reverse=True)[:top_k]
    return ranked


def recommend_precautions(risk_label: str, reasons: list[dict]) -> list[str]:
    precautions = []
    top_features = {item["feature"] for item in reasons}

    if "n" in top_features:
        precautions.append("Balance nitrogen dosing with split application and field-level soil testing.")
    if "p" in top_features or "k" in top_features:
        precautions.append("Apply location-specific phosphorus and potassium corrections before key growth stages.")
    if "ph" in top_features:
        precautions.append("Adjust soil pH using lime or gypsum according to soil test recommendations.")
    if "soil_fertility" in top_features:
        precautions.append("Increase organic matter inputs (compost/green manure) to improve fertility stability.")
    if any(feature in top_features for feature in ["crop", "season", "state", "district"]):
        precautions.append("Prefer crop varieties validated for this location-season combination.")

    if risk_label.upper() == "HIGH":
        precautions.append("Use intensive monitoring and stage-wise nutrient/irrigation corrections this season.")
    elif risk_label.upper() == "MEDIUM":
        precautions.append("Apply preventive nutrient and irrigation adjustments during early growth stages.")
    else:
        precautions.append("Maintain current practices and continue periodic soil and yield monitoring.")

    deduped = []
    for item in precautions:
        if item not in deduped:
            deduped.append(item)
    return deduped[:6]
