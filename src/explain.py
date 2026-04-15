from src.config import NUMERICAL


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


def _format_value(feature: str, value):
    if feature in NUMERICAL:
        return float(value)
    return str(value)


def _reference_value(preprocessor, raw_frame, feature: str):
    state = str(raw_frame.iloc[0]["state"]).strip().lower()
    district = str(raw_frame.iloc[0]["district"]).strip().lower()
    crop = str(raw_frame.iloc[0]["crop"]).strip().lower()

    if feature in NUMERICAL:
        profile = preprocessor._fallback_profile(state=state, district=district)
        return float(profile.get(feature, preprocessor.numeric_medians.get(feature, 0.0))), "location_profile"
    if feature == "season":
        season = preprocessor.infer_season(state=state, district=district, crop=crop)
        return season, "historical_mode"
    return None, "not_applicable"


def _reason_text(feature: str, value, reference, contribution_pct: float) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    influence_text = f"{label} contributed {contribution_pct:.1f}% of local model influence."
    if feature in NUMERICAL and reference is not None:
        delta = float(value) - float(reference)
        direction = "above" if delta > 0 else "below" if delta < 0 else "at"
        return f"{influence_text} Input is {direction} location reference by {abs(delta):.3f}."
    if feature == "season" and reference is not None:
        status = "matches" if str(value).lower() == str(reference).lower() else "differs from"
        return f"{influence_text} Season {status} historical seasonal pattern."
    return influence_text


def explain_prediction(model, preprocessor, encoded_x, raw_frame, features, top_k: int = 5):
    explanation_matrix, _ = model.explain(encoded_x)
    contributions = explanation_matrix[0]
    total = float(contributions.sum()) if float(contributions.sum()) > 0 else 1.0

    ranked = sorted(
        zip(features, contributions),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_k]

    reasons = []
    for feature, contribution in ranked:
        value = raw_frame.iloc[0][feature]
        reference, reference_source = _reference_value(preprocessor, raw_frame, feature)
        contribution_pct = (float(contribution) / total) * 100.0
        reasons.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "value": _format_value(feature, value),
                "reference_value": _format_value(feature, reference) if reference is not None else None,
                "reference_source": reference_source,
                "contribution": float(contribution),
                "contribution_pct": float(contribution_pct),
                "reason": _reason_text(feature, value, reference, float(contribution_pct)),
            }
        )
    return reasons


def recommend_precautions(risk_label: str, reasons: list[dict]) -> list[str]:
    precautions = []
    top_features = {item["feature"] for item in reasons}

    if "n" in top_features:
        precautions.append("Balance nitrogen with split fertilizer application based on soil test.")
    if "p" in top_features or "k" in top_features:
        precautions.append("Use site-specific phosphorus and potassium dosing before the next irrigation.")
    if "ph" in top_features:
        precautions.append("Correct soil pH with lime (for acidic soil) or gypsum/sulfur (for alkaline soil).")
    if "soil_fertility" in top_features:
        precautions.append("Add organic matter or compost to improve soil fertility and nutrient retention.")
    if any(feature in top_features for feature in ["crop", "season", "state", "district"]):
        precautions.append("Prefer crop varieties proven for this location and season.")

    if risk_label.upper() == "HIGH":
        precautions.append("Increase monitoring frequency for soil moisture and nutrient stress through the season.")
    elif risk_label.upper() == "MEDIUM":
        precautions.append("Apply preventive nutrient and irrigation adjustments in the first growth stages.")
    else:
        precautions.append("Maintain current practices and continue periodic soil testing.")

    # Keep output concise and actionable.
    deduped = []
    for item in precautions:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]
