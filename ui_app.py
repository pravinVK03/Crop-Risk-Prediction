import json

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_from_api_payload


@st.cache_resource
def get_bundle():
    return load_model()


def get_dropdown_options(bundle):
    pre = bundle.preprocessor
    states = sorted(pre.category_maps.get("state", {}).keys())
    crops = sorted(pre.category_maps.get("crop", {}).keys())
    seasons = sorted(pre.category_maps.get("season", {}).keys())

    state_to_districts = {}
    for (state, district) in getattr(pre, "location_profiles", {}).keys():
        state_to_districts.setdefault(state, set()).add(district)

    if not state_to_districts:
        all_districts = sorted(pre.category_maps.get("district", {}).keys())
        for state in states:
            state_to_districts[state] = set(all_districts)

    # Optional progressive filters from learned historical profiles.
    state_district_to_crops = {}
    for (state, district, crop), _season in getattr(pre, "season_by_loc_crop", {}).items():
        state_district_to_crops.setdefault((state, district), set()).add(crop)

    state_crop_to_seasons = {}
    for (state, crop), season in getattr(pre, "season_by_state_crop", {}).items():
        state_crop_to_seasons.setdefault((state, crop), set()).add(season)

    normalized = {state: sorted(list(districts)) for state, districts in state_to_districts.items()}
    crops_map = {key: sorted(list(values)) for key, values in state_district_to_crops.items()}
    seasons_map = {key: sorted(list(values)) for key, values in state_crop_to_seasons.items()}
    return states, normalized, crops, seasons, crops_map, seasons_map


def main():
    st.set_page_config(page_title="Crop Risk Inference", page_icon="🌾", layout="centered")
    st.title("Crop Risk Inference")
    st.caption("TabNet inference with real-time inputs. Use dropdowns or manual input.")

    try:
        bundle = get_bundle()
    except Exception as exc:
        st.error("Model loading failed. Train first using `python run_train.py`.")
        st.exception(exc)
        return

    states, state_to_districts, crops, seasons, crops_map, seasons_map = get_dropdown_options(bundle)

    mode = st.radio("Input Mode", ["Dropdown", "Manual"], horizontal=True)

    if mode == "Dropdown":
        selected_state = st.selectbox("State", states, index=0 if states else None)
        district_options = state_to_districts.get(selected_state, [])
        selected_district = st.selectbox(
            "District",
            district_options,
            index=0 if district_options else None,
        )

        crop_options = crops_map.get((selected_state, selected_district), crops)
        selected_crop = st.selectbox("Crop", crop_options, index=0 if crop_options else None)

        state = selected_state or ""
        district = selected_district or ""
        crop = selected_crop or ""
    else:
        state = st.text_input("State", placeholder="e.g. tamilnadu")
        district = st.text_input("District", placeholder="e.g. vellore")
        crop = st.text_input("Crop", placeholder="e.g. rice")

    override = st.checkbox("Override inferred season/soil values")
    payload = {"location": {"state": state.strip(), "district": district.strip()}, "crop_type": crop.strip()}

    if override:
        if mode == "Dropdown":
            season_options = seasons_map.get((state, crop), seasons)
        else:
            season_options = seasons

        season = st.selectbox("Season", season_options, index=0 if season_options else None)
        n_val = st.number_input("N", value=1.0)
        p_val = st.number_input("P", value=1.0)
        k_val = st.number_input("K", value=1.0)
        ph_val = st.number_input("pH", value=7.0)
        fert_val = st.number_input("Soil Fertility", value=1.0)
        payload["season"] = season
        payload["soil"] = {
            "n": float(n_val),
            "p": float(p_val),
            "k": float(k_val),
            "ph": float(ph_val),
            "soil_fertility": float(fert_val),
        }

    submitted = st.button("Predict Risk")

    if not submitted:
        return

    if not state.strip() or not district.strip() or not crop.strip():
        st.error("Please provide State, District, and Crop.")
        return

    try:
        result = predict_from_api_payload(payload, bundle=bundle)
    except Exception as exc:
        st.error("Inference failed for this input.")
        st.exception(exc)
        return

    prediction = result["prediction"]
    input_row = result["input"]
    inferred_fields = result.get("inferred_fields", [])

    st.subheader("Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk", prediction["risk_label"])
    c2.metric("Class", prediction["risk_class"])
    c3.metric("Confidence", f"{prediction['confidence']:.4f}")
    st.caption("Confidence and class probabilities are computed per input (not static).")

    st.subheader("Class Probabilities")
    st.json(prediction.get("class_probabilities", {}))
    st.caption(prediction.get("explanation_method", ""))
    st.caption(prediction.get("explanation_note", ""))

    st.subheader("Inferred Fields")
    if inferred_fields:
        st.write(", ".join(inferred_fields))
    else:
        st.write("None")

    st.subheader("Input Used")
    st.json(input_row)

    st.subheader("Influential Factors")
    reasons = prediction["reasons"]
    table_rows = []
    for item in reasons:
        table_rows.append(
            {
                "feature": item["feature_label"],
                "input_value": item["value"],
                "reference_value": item["reference_value"],
                "reference_source": item["reference_source"],
                "influence_%": round(item["contribution_pct"], 2),
                "influence_score": round(item["contribution"], 6),
                "explanation": item["reason"],
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    st.subheader("Reason Summary")
    for idx, reason in enumerate(reasons, start=1):
        st.write(f"{idx}. {reason['reason']}")

    st.subheader("Precautions")
    for idx, item in enumerate(prediction["precautions"], start=1):
        st.write(f"{idx}. {item}")

    with st.expander("Raw JSON"):
        st.code(json.dumps(result, indent=2), language="json")


if __name__ == "__main__":
    main()
