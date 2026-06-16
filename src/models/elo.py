from typing import Self

import numpy as np
import pandas as pd

from src.models.model import TennisPredictorModel


class TennisPredictorElo(TennisPredictorModel):
    """Predict Player A win probabilities from Elo differences."""

    MODEL_NAME = "tennis_predictor_elo"

    def __init__(self) -> None:
        super().__init__(parametric=False)

    def learn(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        elo_diff = X["global_elo_diff"].to_numpy(dtype=float)

        return 1.0 / (1.0 + np.power(10.0, -elo_diff / 400.0))

    def save(self) -> None:
        pass

    @classmethod
    def load(cls, version: int = None) -> Self:
        return cls()
