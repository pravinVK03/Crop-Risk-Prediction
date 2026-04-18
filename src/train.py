import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CATEGORICAL,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERICAL,
    PREPROCESSOR_PATH,
    TABTRANSFORMER_PARAMS,
    TRAIN_PARAMS,
)
from src.data_loader import load_data
from src.preprocessing import encode_data
from src.tab_transformer.config import ModelConfig
from src.tab_transformer.model import TabTransformerClassifier


def _split_cat_num(x: np.ndarray):
    x_cat = x[:, : len(CATEGORICAL)].astype(np.int64)
    x_num = x[:, len(CATEGORICAL) :].astype(np.float32)
    return x_cat, x_num


def _build_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    x_cat, x_num = _split_cat_num(x)
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x_cat)),
        torch.from_numpy(np.ascontiguousarray(x_num)),
        torch.from_numpy(np.ascontiguousarray(y.astype(np.int64))),
    )


def _evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses = []
    preds, labels = [], []
    with torch.no_grad():
        for x_cat, x_num, y in loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            logits = model(x_cat, x_num)
            loss = criterion(logits, y)
            losses.append(loss.item())
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return float(np.mean(losses)), np.array(preds), np.array(labels)


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

    train_loader = DataLoader(_build_dataset(x_train, y_train), batch_size=TRAIN_PARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(_build_dataset(x_val, y_val), batch_size=TRAIN_PARAMS["batch_size"], shuffle=False)
    test_loader = DataLoader(_build_dataset(x_test, y_test), batch_size=TRAIN_PARAMS["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig(**TABTRANSFORMER_PARAMS)
    model = TabTransformerClassifier(
        cat_cardinalities=preprocessor.get_cat_dims(),
        num_features=len(NUMERICAL),
        num_classes=len(preprocessor.index_to_target),
        config=model_cfg,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_PARAMS["learning_rate"],
        weight_decay=TRAIN_PARAMS["weight_decay"],
    )

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, TRAIN_PARAMS["epochs"] + 1):
        model.train()
        train_losses = []
        skipped_batches = 0
        for x_cat, x_num, y_batch in train_loader:
            x_cat, x_num, y_batch = x_cat.to(device), x_num.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_cat, x_num)
            loss = criterion(logits, y_batch)
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=TRAIN_PARAMS.get("grad_clip_norm", 1.0),
            )
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss, val_pred, val_true = _evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_pred)
        print(
            f"Epoch {epoch:02d}/{TRAIN_PARAMS['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        if skipped_batches:
            print(f"  skipped_non_finite_batches={skipped_batches}")

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= TRAIN_PARAMS["patience"]:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_pred, test_true = _evaluate(model, test_loader, device)
    report = classification_report(test_true, test_pred, output_dict=True, zero_division=0)

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": TABTRANSFORMER_PARAMS,
            "num_features": len(NUMERICAL),
            "cat_cardinalities": preprocessor.get_cat_dims(),
            "num_classes": len(preprocessor.index_to_target),
            "features": FEATURES,
            "categorical": CATEGORICAL,
            "numerical": NUMERICAL,
        },
        MODEL_PATH,
    )
    preprocessor.save(PREPROCESSOR_PATH)

    metrics = {
        "accuracy": float(accuracy_score(test_true, test_pred)),
        "test_loss": float(test_loss),
        "classification_report": report,
        "classes": preprocessor.index_to_target,
        "features": FEATURES,
        "model": "TabTransformer",
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {"model": model, "preprocessor": preprocessor, "metrics": metrics}
