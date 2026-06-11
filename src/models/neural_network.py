from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 25
HIDDEN_DIMS = (64, 32)
DROPOUT = 0.2


class TennisPredictorMLP(nn.Module):
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


def learn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[TennisPredictorMLP, Dict[str, object]]:
    """Train a binary classifier to predict whether Player A wins a match."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert training data to PyTorch tensors and create DataLoader
    X_tensor = torch.tensor(X_train.to_numpy(dtype="float32"), dtype=torch.float32)
    y_tensor = torch.tensor(y_train.to_numpy(dtype="float32"), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Configure neural network, loss function, and optimizer
    model = TennisPredictorMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for batch_features, batch_targets in loop:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    return model
