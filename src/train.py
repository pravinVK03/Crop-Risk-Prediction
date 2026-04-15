import json
from pathlib import Path

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL,
    METRICS_PATH,
    MODEL_BASENAME,
    NUMERICAL,
    PREPROCESSOR_PATH,
    TABNET_PARAMS,
)
from src.data_loader import load_data
from src.preprocessing import encode_data


def train_model(df=None):
    frame = load_data() if df is None else df
    x, y, preprocessor = encode_data(frame)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    cat_idxs = list(range(len(CATEGORICAL)))
    cat_dims = preprocessor.get_cat_dims()

    model = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=8,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 2e-3},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 15, "gamma": 0.9},
        **TABNET_PARAMS,
        verbose=1,
    )

    model.fit(
        X_train=x_train,
        y_train=y_train,
        eval_set=[(x_val, y_val)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=80,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
    )

    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    Path(MODEL_BASENAME).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_BASENAME)
    preprocessor.save(PREPROCESSOR_PATH)

    metrics = {
        "accuracy": float(np.mean(predictions == y_test)),
        "classification_report": report,
        "classes": preprocessor.index_to_target,
        "features": CATEGORICAL + NUMERICAL,
    }

    with Path(METRICS_PATH).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {"model": model, "preprocessor": preprocessor, "metrics": metrics}
