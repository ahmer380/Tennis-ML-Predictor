from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models.base import TennisPredictorModel

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
HIDDEN_DIMS = (64, 32)
DROPOUT = 0.2


class _MLPNetwork(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TennisPredictorMLP(TennisPredictorModel):
    """Predict Player A win probabilities using a Multi-Layer Perceptron (MLP) neural network."""

    MODEL_NAME = "TennisPredictorMLP"

    def __init__(self) -> None:
        super().__init__()
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
        # Convert training and validation data to normalised PyTorch tensors and create DataLoaders
        X_tensor = torch.tensor(self._scaler.fit_transform(X_train), dtype=torch.float32)
        y_tensor = torch.tensor(y_train.to_numpy(dtype="float32"), dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        X_validation_tensor = torch.tensor(self._scaler.transform(X_validation), dtype=torch.float32).to(self._device)
        y_validation_tensor = torch.tensor(y_validation.to_numpy(dtype="float32"), dtype=torch.float32).to(self._device)

        # Configure neural network, loss function, and optimizer
        self._network = _MLPNetwork(input_dim=X_train.shape[1]).to(self._device)
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
