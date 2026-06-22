from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.model import TennisPredictorModel


def evaluate_model(
    model: TennisPredictorModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_data: bool = True,
) -> None:
    """Evaluate the model on the test set and print/save evaluation metrics and plots."""
    print(f"\nEvaluating {model.instance_name}...")

    y_prob = model.predict(X_test)
    y_pred = model.predict_class(X_test)

    if save_data:
        _plot_confusion_matrix(model, y_test, y_pred)
        _plot_prediction_histogram(model, y_prob)
        _save_evaluation_metrics(model, y_test, y_pred, y_prob)

    print(f"Log Loss:      {_format_metric(log_loss(y_test, y_prob))}")
    print(f"Brier Score:   {_format_metric(brier_score_loss(y_test, y_prob))}")
    print(f"ROC AUC:       {_format_metric(roc_auc_score(y_test, y_prob))}")
    print()
    print(f"Accuracy:      {_format_metric(accuracy_score(y_test, y_pred))}")
    print(f"Precision:     {_format_metric(precision_score(y_test, y_pred, zero_division=0))}")
    print(f"Recall:        {_format_metric(recall_score(y_test, y_pred, zero_division=0))}")
    print(f"F1 Score:      {_format_metric(f1_score(y_test, y_pred, zero_division=0))}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    if save_data:
        print("Evaluation plots saved to:")
        print(f"{model.instance_dir}/confusion_matrix.png")
        print(f"{model.instance_dir}/prediction_histogram.png")
        print(f"{model.instance_dir}/evaluation_metrics.txt")
    print()


def _plot_confusion_matrix(model: TennisPredictorModel, y_true: pd.Series, y_pred: np.ndarray) -> None:
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    confusion_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Player 0", "Player 1"])

    fig, ax = plt.subplots(figsize=(5, 4))
    confusion_display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model.instance_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(model.instance_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_prediction_histogram(model: TennisPredictorModel, y_prob: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob, bins=20, range=(0.0, 1.0), color="#4C72B0", edgecolor="white")
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{model.instance_name} Prediction Histogram")
    fig.tight_layout()
    fig.savefig(model.instance_dir / "prediction_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_evaluation_metrics(
    model: TennisPredictorModel, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray
) -> None:
    metrics = {
        "Log Loss": log_loss(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }

    with open(model.instance_dir / "evaluation_metrics.txt", "w") as f:
        f.write(f"Evaluation Metrics for {model.instance_name}\n\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {_format_metric(metric_value)}\n")


def _format_metric(value: Any) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)
