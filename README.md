# Crop Risk Prediction (Application-Oriented, TabTransformer)

This project is a **tabular crop risk decision-support application**.
It predicts crop risk as `LOW`, `MEDIUM`, or `HIGH` using historical crop + soil data and provides explainable reasons and precautions.

## Application Goal

Enable practical crop-risk screening with minimal user input:

- `State`
- `District`
- `Crop`

The system infers missing context (season/soil defaults from historical profiles), predicts risk, and explains the prediction in plain language.

## End-User Workflow

1. User enters location and crop.
2. System prepares complete model input by filling missing fields.
3. TabTransformer predicts risk class and confidence.
4. XAI module identifies influential factors.
5. App outputs precautions linked to those factors.

## Input and Output

Input (required):
- `state`, `district`, `crop`

Optional override:
- `season`, `n`, `p`, `k`, `ph`, `soil_fertility`, `year`, `area`, `production`, `yield_log`

Output:
- risk label (`LOW`/`MEDIUM`/`HIGH`)
- confidence and class probabilities
- influential factors (value, reference, contribution)
- precautions/recommendations
- list of inferred fields

## Key Application Features

- Dynamic inference per user input (not static templates)
- Progressive UI selections (districts based on selected state)
- Explainable AI with per-feature contribution analysis
- Transparent reference values used for explanations
- Metrics + visual analytics generation for reporting

## Quick Start

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Train model:

```powershell
python run_train.py
```

3. CLI inference:

```powershell
python run_infer.py
```

4. Metrics and plots:

```powershell
python -u calculate_metrics.py
```

5. Streamlit app:

```powershell
python -m streamlit run ui_app.py
```

## Files You Will Use Most

- training: `run_train.py`
- inference: `run_infer.py`
- UI: `ui_app.py`
- metrics + graphs: `calculate_metrics.py`
- model code: `src/train.py`, `src/predict.py`, `src/explain.py`

## Model and Artifacts

- model: `models/tab_transformer/crop_risk_tab_transformer.pt`
- preprocessor: `models/tab_transformer/preprocessor.pkl`
- training metrics: `models/tab_transformer/metrics.json`
- performance report: `outputs/performance_metrics.json`
- parameter stats: `outputs/parameter_statistics.json`
- visual outputs: `outputs/metrics_plots/`

## Feature Set

Categorical:
- `state`, `district`, `season`, `crop`

Numerical:
- `year`, `area`, `production`, `yield_log`, `n`, `p`, `k`, `ph`, `soil_fertility`

## Explainability Method

The application uses perturbation-based local attribution:

1. Keep input fixed.
2. Replace one feature with its reference value.
3. Recompute prediction probability.
4. Measure contribution from probability change.

This gives transparent, case-specific reasons for each prediction.

## Deployment Note

Current scope is **tabular-only** and ready for application use.
Satellite and IoT branches are intentionally excluded in this cleaned version.
