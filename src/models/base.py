from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


class TennisPredictorModel(ABC):
    MODEL_NAME = "TennisPredictorModel"

    def __init__(self) -> None:
        model_artifacts_dir = ARTIFACTS_DIR / self.MODEL_NAME
        model_artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.version = len(list(model_artifacts_dir.glob(f"v*"))) + 1
        self.instance_name = f"{self.MODEL_NAME}_v{self.version}"
        self.instance_dir = model_artifacts_dir / f"v{self.version}"
        self.instance_dir.mkdir(exist_ok=False)

    @abstractmethod
    def learn(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict(X) >= 0.5).astype(int)
