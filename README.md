# Crop Risk Prediction Framework (TabTransformer Only)

This repository is now preserved as a tabular-only crop risk system using `TabTransformer`.

## Quick Start

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Train model:

```powershell
python run_train.py
```

3. Infer from `State + District + Crop`:

```powershell
python run_infer.py
```

4. Calculate metrics and plots:

```powershell
python -u calculate_metrics.py
```

5. Launch UI:

```powershell
python -m streamlit run ui_app.py
```

## Model Artifacts

- model: `models/tab_transformer/crop_risk_tab_transformer.pt`
- preprocessor: `models/tab_transformer/preprocessor.pkl`
- training metrics: `models/tab_transformer/metrics.json`
- eval metrics: `outputs/performance_metrics.json`
- eval plots: `outputs/metrics_plots/`

## Features Used

- categorical: `state, district, season, crop`
- numerical: `year, area, production, yield_log, n, p, k, ph, soil_fertility`

## Explainability

The system returns dynamic local explanations using perturbation-based attribution:

- influential features
- reference values and sources
- influence percentage and direction
- precautions linked to top contributing factors
