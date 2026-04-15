import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.predict import load_model


def risk_label(raw_target: int) -> str:
    mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return mapping.get(int(raw_target), f"RISK_{raw_target}")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate performance metrics for trained TabNet model.")
    parser.add_argument(
        "--output",
        default="outputs/performance_metrics.json",
        help="Path to save computed metrics JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_model()
    frame = load_data()

    x, y = bundle.preprocessor.transform(frame, with_target=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    # Keep the same extra split pattern used in training.
    train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    y_pred = bundle.model.predict(x_test)
    y_proba = bundle.model.predict_proba(x_test)
    class_indices = sorted(bundle.preprocessor.index_to_target.keys())

    cm_array = confusion_matrix(y_test, y_pred, labels=class_indices)
    matrix = cm_array.tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    roc_auc = float(
        roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted", labels=class_indices)
    )

    named_classes = {
        str(index): {
            "raw_class": int(bundle.preprocessor.index_to_target[index]),
            "risk_label": risk_label(bundle.preprocessor.index_to_target[index]),
        }
        for index in class_indices
    }
    class_labels = [named_classes[str(index)]["risk_label"] for index in class_indices]

    contingency_table = {}
    for actual_i, actual_label in enumerate(class_labels):
        contingency_table[actual_label] = {}
        for pred_i, pred_label in enumerate(class_labels):
            contingency_table[actual_label][pred_label] = int(cm_array[actual_i, pred_i])

    metrics = {
        "num_samples_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": class_labels,
        "contingency_table_actual_vs_predicted": contingency_table,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Metrics saved to: {output_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Confusion Matrix (Actual x Predicted):")
    labels = metrics["confusion_matrix_labels"]
    print("Actual\\Predicted\t" + "\t".join(labels))
    for actual_label, row in zip(labels, metrics["confusion_matrix"]):
        print(actual_label + "\t\t" + "\t".join(str(value) for value in row))


if __name__ == "__main__":
    main()
