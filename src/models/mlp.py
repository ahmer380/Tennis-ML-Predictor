from typing import Optional, Self
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss
import joblib

from src.models.model import TennisPredictorModel

BATCH_SIZE = 256
LEARNING_RATE = 3e-4
EPOCHS = 250
HIDDEN_DIMS = (128, 64, 32)
DROPOUTS = (0.4, 0.3, 0.1)
PATIENCE = 25


class _MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int], dropouts: tuple[float]) -> None:
        super().__init__()

        layers = []
        previous_dim = input_dim

        for dropout, hidden_dim in zip(dropouts, hidden_dims):
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))  # Output layer for binary classification

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TennisPredictorMLP(TennisPredictorModel):
    """Predict Player A win probabilities using a Multi-Layer Perceptron (MLP) neural network."""

    MODEL_NAME = "tennis_predictor_mlp"

    def __init__(self, version: int = None) -> None:
        super().__init__(parametric=True, version=version)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._scaler = StandardScaler()
        self._network: Optional[_MLPNetwork] = None
        self._is_fitted = False

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

        # Convert training and validation data to normalised PyTorch tensors and create DataLoaders
        X_tensor = torch.tensor(self._scaler.fit_transform(X_train), dtype=torch.float32)
        y_tensor = torch.tensor(y_train.to_numpy(dtype="float32"), dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        X_validation_tensor = torch.tensor(self._scaler.transform(X_validation), dtype=torch.float32).to(self._device)
        y_validation_tensor = torch.tensor(y_validation.to_numpy(dtype="float32"), dtype=torch.float32).to(self._device)

        # Configure neural network, loss function, and optimizer
        self._network = _MLPNetwork(input_dim=X_train.shape[1], hidden_dims=HIDDEN_DIMS, dropouts=DROPOUTS).to(
            self._device
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self._network.parameters(), lr=LEARNING_RATE)

        best_validation_loss = float("inf")
        patience_counter = 0
        best_model_state_dict = None

        for epoch in range(EPOCHS):
            # Train the model for one epoch
            self._network.train()

            train_logits = []
            train_targets = []

            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self._device)
                batch_targets = batch_targets.to(self._device)

                optimizer.zero_grad()
                logits = self._network(batch_features)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

                train_logits.append(logits.detach().cpu())
                train_targets.append(batch_targets.detach().cpu())

            train_probabilities = torch.sigmoid(torch.cat(train_logits))
            train_class_predictions = (train_probabilities >= 0.5).int()

            train_loss = criterion(torch.cat(train_logits), torch.cat(train_targets)).item()
            train_accuracy = accuracy_score(torch.cat(train_targets), train_class_predictions)
            train_brier_score = brier_score_loss(torch.cat(train_targets), train_probabilities)

            # Evaluate the model on the validation set
            self._network.eval()
            with torch.no_grad():
                validation_logits = self._network(X_validation_tensor)
                validation_probabilities = torch.sigmoid(validation_logits).cpu().numpy()
                validation_class_predictions = (validation_probabilities >= 0.5).astype(int)

                validation_loss = criterion(validation_logits, y_validation_tensor).item()
                validation_accuracy = accuracy_score(y_validation_tensor.cpu().numpy(), validation_class_predictions)
                validation_brier_score = brier_score_loss(y_validation_tensor.cpu().numpy(), validation_probabilities)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | ",
                f"Train Loss: {train_loss:.4f} - Acc: {train_accuracy * 100:.1f}% - Brier: {train_brier_score:.4f} | ",
                f"Val Loss: {validation_loss:.4f} - Acc: {validation_accuracy * 100:.1f}% - Brier: {validation_brier_score:.4f}",
            )

            # Early stopping
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                patience_counter = 0
                best_model_state_dict = copy.deepcopy(self._network.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        self._network.load_state_dict(best_model_state_dict)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("TennisPredictorMLP must be trained before calling predict().")

        X_tensor = torch.tensor(self._scaler.transform(X), dtype=torch.float32).to(self._device)

        self._network.eval()
        with torch.no_grad():
            logits = self._network(X_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        return np.asarray(probabilities, dtype=float)

    def save(self) -> None:
        if not self._is_fitted or self._network is None:
            raise RuntimeError("TennisPredictorMLP must be trained before calling save().")

        checkpoint = {
            "model_state_dict": self._network.state_dict(),
            "input_dim": self._network.input_dim,
            "hidden_dims": self._network.hidden_dims,
            "dropouts": self._network.dropouts,
        }
        torch.save(checkpoint, self.instance_dir / "model.pth")
        joblib.dump(self._scaler, self.instance_dir / "scaler.joblib")

    @classmethod
    def load(cls, version: int = None) -> Self:
        model_instance = cls(version=version)

        # Load the model state dictionary and scaler
        checkpoint = torch.load(model_instance.instance_dir / "model.pth")
        model_instance._network = _MLPNetwork(
            input_dim=checkpoint["input_dim"], hidden_dims=checkpoint["hidden_dims"], dropouts=checkpoint["dropouts"]
        ).to(model_instance._device)
        model_instance._network.load_state_dict(checkpoint["model_state_dict"])
        model_instance._scaler = joblib.load(model_instance.instance_dir / "scaler.joblib")
        model_instance._is_fitted = True

        return model_instance
