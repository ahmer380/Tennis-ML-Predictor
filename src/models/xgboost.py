from typing import Self

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from src.models.model import TennisPredictorModel


class TennisPredictorXGBoost(TennisPredictorModel):
    """Predict Player A win probabilities using an XGBoost Classifier."""

    MODEL_NAME = "tennis_predictor_xgboost"

    def __init__(self, version: int = None) -> None:
        super().__init__(trainable=True, version=version)
        self._is_fitted = False
        self._xgbclassifier = XGBClassifier()

    def learn(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> None:
        assert (
            not self._is_fitted
        ), f"{self.instance_name} has already been trained. Create a new instance to train again."

        print(f"\nTraining {self.MODEL_NAME}...")

        # Configure XGBoost Classifier
        self._xgbclassifier = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=100,
            eval_metric=["logloss", "error"],
        )

        # Train the model
        self._xgbclassifier.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_validation, y_validation)],
            verbose=50,
        )
        self._is_fitted = True

        # Evaluate the model on the validation set
        val_probabilities = self.predict(X_validation)
        val_class_predictions = self.predict_class(X_validation)
        y_val_np = y_validation.to_numpy()
        val_log_loss = log_loss(y_val_np, val_probabilities)
        val_acc = accuracy_score(y_val_np, val_class_predictions)
        val_brier = brier_score_loss(y_val_np, val_probabilities)

        print("--- Training Complete ---")
        print(f"Best Iteration: {self._xgbclassifier.best_iteration}")
        print(
            f"Validation Log Loss: {val_log_loss:.4f} | Validation Accuracy: {val_acc * 100:.2f}% | Validation Brier Score: {val_brier:.4f}"
        )
        self.log_feature_importance(feature_names=X_train.columns.tolist())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"{self.instance_name} must be trained before calling predict().")

        probabilities = self._xgbclassifier.predict_proba(X)[:, 1]

        return np.asarray(probabilities, dtype=float)

    def save(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.instance_name} must be trained before calling save().")

        self._xgbclassifier.save_model(self.instance_dir / "model.json")

    @classmethod
    def load(cls, version: int = None) -> Self:
        model_instance = cls(version=version)

        model_instance._xgbclassifier.load_model(model_instance.instance_dir / "model.json")
        model_instance._is_fitted = True

        return model_instance

    def log_feature_importance(self, feature_names: list[str]) -> None:
        """Log feature importance of the trained XGBoost model as a bar chart image."""

        if not self._is_fitted:
            raise RuntimeError(f"{self.instance_name} must be trained before calling log_feature_importance().")

        importances = self._xgbclassifier.feature_importances_
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Length of feature_names ({len(feature_names)}) does not match number of features in the model ({len(importances)})."
            )
        feature_importance_pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        names, scores = zip(*feature_importance_pairs)

        plt.figure(figsize=(10, 6))
        plt.barh(names, scores)
        plt.gca().invert_yaxis()
        plt.xlabel("Feature Importance")
        plt.title(f"Top 10 Feature Importances - {self.instance_name}")

        plt.tight_layout()

        # Save as image instead of text
        output_path = self.instance_dir / "top_10_feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Feature importance plot saved to: {output_path}")
