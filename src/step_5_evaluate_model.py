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
    roc_curve,
)

from src.models.model import TennisPredictorModel


def plot_confusion_matrix(model: TennisPredictorModel, y_true: pd.Series, y_pred: np.ndarray) -> None:
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    confusion_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Player 0", "Player 1"])

    fig, ax = plt.subplots(figsize=(5, 4))
    confusion_display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel(f"Predicted label\n\nAccuracy: {accuracy_score(y_true, y_pred):.3f}")
    ax.set_title(f"{model.instance_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(model.instance_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(model: TennisPredictorModel, y_true: pd.Series, y_prob: np.ndarray) -> None:
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model.instance_name} ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(model.instance_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_histogram(model: TennisPredictorModel, y_prob: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob, bins=20, range=(0.0, 1.0), color="#4C72B0", edgecolor="white")
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{model.instance_name} Prediction Histogram")
    fig.tight_layout()
    fig.savefig(model.instance_dir / "prediction_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_metric(value: Any) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def evaluate_model(
    model: TennisPredictorModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_plots: bool = True,
) -> None:
    y_prob = model.predict(X_test)
    y_pred = model.predict_class(X_test)

    if save_plots:
        plot_confusion_matrix(model, y_test, y_pred)
        plot_roc_curve(model, y_test, y_prob)
        plot_prediction_histogram(model, y_prob)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Model:         {model.instance_name}")
    print()
    print(f"Accuracy:      {format_metric(accuracy_score(y_test, y_pred))}")
    print(f"Precision:     {format_metric(precision_score(y_test, y_pred, zero_division=0))}")
    print(f"Recall:        {format_metric(recall_score(y_test, y_pred, zero_division=0))}")
    print(f"F1 Score:      {format_metric(f1_score(y_test, y_pred, zero_division=0))}")
    print()
    print(f"Log Loss:      {format_metric(log_loss(y_test, y_prob))}")
    print(f"Brier Score:   {format_metric(brier_score_loss(y_test, y_prob))}")
    print(f"ROC AUC:       {format_metric(roc_auc_score(y_test, y_prob))}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print()
    print("Plots saved to:")
    print(f"{model.instance_dir}/")
    print("=" * 50)
