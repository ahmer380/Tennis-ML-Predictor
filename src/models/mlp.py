from typing import Optional, Self

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models.model import TennisPredictorModel

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10


class _MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int], dropout: float) -> None:
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))  # Output layer for binary classification

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

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
        self._network = _MLPNetwork(input_dim=X_train.shape[1], hidden_dims=(64, 32), dropout=0.2).to(self._device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self._network.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS + 1):
            # Train the model for one epoch
            self._network.train()

            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            if epoch != 0:
                for batch_features, batch_targets in dataloader:
                    batch_features = batch_features.to(self._device)
                    batch_targets = batch_targets.to(self._device)
                    batch_size = batch_targets.size(dim=0)

                    optimizer.zero_grad()
                    logits = self._network(batch_features)
                    loss = criterion(logits, batch_targets)
                    loss.backward()
                    optimizer.step()

                    train_loss_sum += loss.item() * batch_size
                    train_correct += ((torch.sigmoid(logits) >= 0.5).float() == batch_targets).sum().item()
                    train_total += batch_size

            train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0

            # Evaluate the model on the validation set
            self._network.eval()
            with torch.no_grad():
                logits = self._network(X_validation_tensor)
                validation_loss = criterion(logits, y_validation_tensor).item()
                validation_accuracy = (
                    ((torch.sigmoid(logits) >= 0.5).float() == y_validation_tensor).sum().item()
                    / y_validation_tensor.size(dim=0)
                    if y_validation_tensor.size(dim=0) > 0
                    else 0.0
                )

            print(
                f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy * 100:.1f}% | "
                f"Val Loss: {validation_loss:.4f} | Val Acc: {validation_accuracy * 100:.1f}%"
            )

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
            "dropout": self._network.dropout,
        }
        torch.save(checkpoint, self.instance_dir / "model.pth")
        torch.save(self._scaler, self.instance_dir / "scaler.pth")

    @classmethod
    def load(cls, version: int = None) -> Self:
        model_instance = cls(version=version)

        # Load the model state dictionary and scaler
        checkpoint = torch.load(model_instance.instance_dir / "model.pth")
        model_instance._network = _MLPNetwork(
            input_dim=checkpoint["input_dim"], hidden_dims=checkpoint["hidden_dims"], dropout=checkpoint["dropout"]
        ).to(model_instance._device)
        model_instance._network.load_state_dict(checkpoint["model_state_dict"])
        model_instance._scaler = torch.load(model_instance.instance_dir / "scaler.pth", weights_only=False)
        model_instance._is_fitted = True

        return model_instance
