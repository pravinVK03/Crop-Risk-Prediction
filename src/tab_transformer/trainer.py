import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader

from .config import DEFAULT_CONFIG, PipelineConfig
from .data import TabPreprocessor, build_splits, load_frame
from .model import TabTransformerClassifier


def _to_device(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device):
    x_cat, x_num, y = batch
    return x_cat.to(device), x_num.to(device), y.to(device)


def _evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            x_cat, x_num, y = _to_device(batch, device)
            logits = model(x_cat, x_num)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    average_loss = total_loss / len(dataloader.dataset)
    return average_loss, np.array(predictions), np.array(labels)


def train_pipeline(config: PipelineConfig = DEFAULT_CONFIG) -> dict:
    frame = load_frame(config.data.data_path)
    preprocessor = TabPreprocessor(
        categorical_columns=config.data.categorical_columns,
        numerical_columns=config.data.numerical_columns,
        target=config.data.target,
    )

    split_data = build_splits(frame=frame, preprocessor=preprocessor, config=config.data)
    train_dataset = preprocessor.to_torch_dataset(*split_data.train)
    val_dataset = preprocessor.to_torch_dataset(*split_data.val)
    test_dataset = preprocessor.to_torch_dataset(*split_data.test)

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabTransformerClassifier(
        cat_cardinalities=preprocessor.categorical_cardinalities,
        num_features=len(config.data.numerical_columns),
        num_classes=preprocessor.num_classes,
        config=config.model,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            x_cat, x_num, y = _to_device(batch, device)
            optimizer.zero_grad()
            logits = model(x_cat, x_num)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_pred, val_true = _evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_pred)
        print(
            f"Epoch {epoch:02d}/{config.train.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.train.early_stop_patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_pred, test_true = _evaluate(model, test_loader, device)
    report = classification_report(test_true, test_pred, output_dict=True, zero_division=0)
    print(f"Test loss: {test_loss:.4f}")
    print(classification_report(test_true, test_pred, zero_division=0))

    artifact_dir = Path(config.paths.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "cat_cardinalities": preprocessor.categorical_cardinalities,
        "num_features": len(config.data.numerical_columns),
        "num_classes": preprocessor.num_classes,
        "model_config": config.model.__dict__,
        "data_config": config.data.__dict__,
    }
    torch.save(checkpoint, config.paths.checkpoint_path)
    preprocessor.save(config.paths.preprocessor_path)

    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy_score(test_true, test_pred),
        "classification_report": report,
    }
    with config.paths.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics
