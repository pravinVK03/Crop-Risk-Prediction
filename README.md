# Tabular Crop Risk (TabNet)

This implementation covers only the tabular branch: crop production + soil properties + location + season.
The model is `TabNetClassifier` and returns:
- risk class (`LOW`, `MEDIUM`, `HIGH`)
- top explainable reasons (feature contributions)
- precaution recommendations

Training uses historical crop production and soil records (`Dataset_Z.csv`).
Inference is real-time friendly: you can provide only crop + location, and the system infers season and soil/production defaults from historical profiles for that location.

## Structure

```text
Agri_Data/
  data/processed/Dataset_Z.csv
  models/tabnet/
    crop_risk_tabnet.zip
    preprocessor.pkl
    metrics.json
  src/
    config.py
    data_loader.py
    preprocessing.py
    train.py
    predict.py
    explain.py
  main.py
  requirements.txt
```

## Run

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Train and run sample inference:

```powershell
python main.py
```

Or run them separately:

```powershell
python run_train.py
python run_infer.py --state "karnataka" --district "mysore" --crop "rice"
```

## Basic UI

```powershell
streamlit run ui_app.py
```

Then open the shown local URL in your browser, enter:
- State
- District
- Crop

The UI will show risk prediction, confidence, inferred fields, reasons, and precautions.
You can choose `Dropdown` or `Manual` input mode and optionally override inferred season/soil values.

Artifacts are saved to `models/tabnet/`.

## API-style Input (minimal)

Minimum input:

```json
{
  "location": { "state": "karnataka", "district": "mysore" },
  "crop_type": "rice"
}
```

You can still optionally pass `season` or `soil` values to override inferred defaults.

## Full API-style Input

`predict_from_api_payload(payload)` expects:

```json
{
  "location": { "state": "karnataka", "district": "mysore" },
  "crop_type": "rice",
  "season": "kharif",
  "soil": { "n": 1.1, "p": 0.95, "k": 1.05, "ph": 6.4, "soil_fertility": 1.0 }
}
```

If any soil/production fields are missing, the preprocessor infers fallback values using historical medians from the same district/state.
If `season` is missing, it is inferred from historical mode for `(state, district, crop)` with fallbacks.
