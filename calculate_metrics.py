import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from src.config import NUMERICAL
from src.data_loader import load_data
from src.predict import load_model, predict_batch


def risk_label(raw_target: int) -> str:
    mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return mapping.get(int(raw_target), f"RISK_{raw_target}")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate performance metrics for trained TabTransformer model.")
    parser.add_argument(
        "--output",
        default="outputs/performance_metrics.json",
        help="Path to save computed metrics JSON.",
    )
    parser.add_argument(
        "--plots-dir",
        default="outputs/metrics_plots",
        help="Directory to save metric visualizations.",
    )
    parser.add_argument(
        "--stats-output",
        default="outputs/parameter_statistics.json",
        help="Path to save feature-level statistical summary JSON.",
    )
    return parser.parse_args()


def save_confusion_heatmap(cm: np.ndarray, labels: list[str], path: Path) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_normalized_confusion_heatmap(cm: np.ndarray, labels: list[str], path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=float)
    np.divide(cm, np.maximum(row_sums, 1), where=row_sums != 0, out=cm_norm)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_metrics_bar(metrics: dict, path: Path) -> None:
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    vals = [metrics[key] for key in keys]
    pretty = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(pretty, vals, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"])
    plt.ylim(0, 1.0)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    for bar, score in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_per_class_metrics_bar(per_class_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(per_class_df))
    width = 0.25
    plt.bar(x - width, per_class_df["precision"], width=width, label="Precision")
    plt.bar(x, per_class_df["recall"], width=width, label="Recall")
    plt.bar(x + width, per_class_df["f1_score"], width=width, label="F1 Score")
    plt.xticks(x, per_class_df["label"].tolist())
    plt.ylim(0, 1.0)
    plt.title("Per-Class Metrics")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_class_distribution(y_true: np.ndarray, labels: list[str], path: Path) -> None:
    counts = [(y_true == idx).sum() for idx in range(len(labels))]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, counts, color=["#2ca02c", "#ff7f0e", "#d62728"])
    plt.title("Test Set Class Distribution")
    plt.ylabel("Samples")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count, str(int(count)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, labels: list[str], path: Path) -> None:
    y_true_bin = label_binarize(y_true, classes=list(range(len(labels))))
    plt.figure(figsize=(8, 6))
    for class_idx, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_proba[:, class_idx])
        auc_val = roc_auc_score(y_true_bin[:, class_idx], y_proba[:, class_idx])
        plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("ROC Curves by Class")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, labels: list[str], path: Path) -> None:
    y_true_bin = label_binarize(y_true, classes=list(range(len(labels))))
    plt.figure(figsize=(8, 6))
    for class_idx, label in enumerate(labels):
        precision_cls, recall_cls, _ = precision_recall_curve(y_true_bin[:, class_idx], y_proba[:, class_idx])
        plt.plot(recall_cls, precision_cls, linewidth=2, label=label)

    plt.title("Precision-Recall Curves by Class")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_probability_histogram(y_proba: np.ndarray, labels: list[str], path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for class_idx, label in enumerate(labels):
        sns.kdeplot(y_proba[:, class_idx], label=label, fill=False, linewidth=2)
    plt.title("Predicted Probability Distribution by Class")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_feature_correlation_heatmap(frame: pd.DataFrame, path: Path) -> None:
    corr = frame[NUMERICAL].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, cbar=True)
    plt.title("Numerical Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_feature_boxplots(frame: pd.DataFrame, path: Path) -> None:
    risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    plot_df = frame.copy()
    plot_df["risk_label"] = plot_df["risk"].map(risk_map).fillna(plot_df["risk"].astype(str))

    n_cols = 3
    n_rows = int(np.ceil(len(NUMERICAL) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, feature in enumerate(NUMERICAL):
        sns.boxplot(data=plot_df, x="risk_label", y=feature, ax=axes[idx])
        axes[idx].set_title(f"{feature} by Risk Class")
        axes[idx].set_xlabel("Risk")
        axes[idx].set_ylabel(feature)

    for idx in range(len(NUMERICAL), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def build_parameter_statistics(frame_test: pd.DataFrame, per_class_df: pd.DataFrame) -> dict:
    overall = frame_test[NUMERICAL].describe().to_dict()
    by_risk = {}
    for raw_class in sorted(frame_test["risk"].dropna().unique().tolist()):
        class_frame = frame_test[frame_test["risk"] == raw_class]
        by_risk[str(int(raw_class))] = {
            "label": risk_label(int(raw_class)),
            "count": int(len(class_frame)),
            "stats": class_frame[NUMERICAL].describe().to_dict(),
        }

    return {
        "overall_numerical_stats": overall,
        "numerical_stats_by_risk": by_risk,
        "per_class_performance": per_class_df.to_dict(orient="records"),
    }


def main():
    args = parse_args()
    bundle = load_model()
    frame = load_data()

    frame_train, frame_test = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["risk"],
    )
    # Keep the same extra split pattern used in training.
    train_test_split(
        frame_train,
        test_size=0.2,
        random_state=42,
        stratify=frame_train["risk"],
    )

    y_pred, y_proba = predict_batch(bundle, frame_test)
    y_test = bundle.preprocessor.transform(frame_test, with_target=True)[1]
    class_indices = sorted(bundle.preprocessor.index_to_target.keys())

    cm_array = confusion_matrix(y_test, y_pred, labels=class_indices)
    matrix = cm_array.tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=class_indices,
        average=None,
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

    per_class_df = pd.DataFrame(
        {
            "class_index": class_indices,
            "label": class_labels,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1_score": per_class_f1,
            "support": per_class_support,
        }
    )

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
        "per_class_metrics": per_class_df.to_dict(orient="records"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    stats = build_parameter_statistics(frame_test=frame_test, per_class_df=per_class_df)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

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
    print(f"Parameter statistics saved to: {stats_path}")

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "confusion_matrix_heatmap.png"
    cm_norm_path = plots_dir / "confusion_matrix_normalized_heatmap.png"
    metric_bar_path = plots_dir / "metrics_bar_chart.png"
    per_class_bar_path = plots_dir / "per_class_metrics_bar_chart.png"
    class_dist_path = plots_dir / "class_distribution.png"
    roc_path = plots_dir / "roc_curves.png"
    pr_path = plots_dir / "precision_recall_curves.png"
    prob_hist_path = plots_dir / "predicted_probability_distribution.png"
    corr_path = plots_dir / "feature_correlation_heatmap.png"
    boxplot_path = plots_dir / "feature_boxplots_by_risk.png"

    save_confusion_heatmap(cm_array, class_labels, cm_path)
    save_normalized_confusion_heatmap(cm_array, class_labels, cm_norm_path)
    save_metrics_bar(metrics, metric_bar_path)
    save_per_class_metrics_bar(per_class_df, per_class_bar_path)
    save_class_distribution(y_test, class_labels, class_dist_path)
    save_roc_curve(y_test, y_proba, class_labels, roc_path)
    save_pr_curve(y_test, y_proba, class_labels, pr_path)
    save_probability_histogram(y_proba, class_labels, prob_hist_path)
    save_feature_correlation_heatmap(frame_test, corr_path)
    save_feature_boxplots(frame_test, boxplot_path)

    print(f"Saved confusion matrix heatmap: {cm_path}")
    print(f"Saved normalized confusion matrix heatmap: {cm_norm_path}")
    print(f"Saved metrics bar chart: {metric_bar_path}")
    print(f"Saved per-class metrics bar chart: {per_class_bar_path}")
    print(f"Saved class distribution chart: {class_dist_path}")
    print(f"Saved ROC curves: {roc_path}")
    print(f"Saved Precision-Recall curves: {pr_path}")
    print(f"Saved predicted probability distribution: {prob_hist_path}")
    print(f"Saved feature correlation heatmap: {corr_path}")
    print(f"Saved feature boxplots by risk class: {boxplot_path}")


if __name__ == "__main__":
    main()
