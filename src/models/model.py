from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


class TennisPredictorModel(ABC):
    MODEL_NAME = "tennis_predictor_model"

    def __init__(self, parametric: bool, version: int = None) -> None:
        model_artifacts_dir = ARTIFACTS_DIR / self.MODEL_NAME
        model_artifacts_dir.mkdir(parents=True, exist_ok=True)

        if parametric:
            if version:
                assert (model_artifacts_dir / f"v{version}").exists(), f"Version {version} for {self.MODEL_NAME} does not exist."
                self.version = version
            else:
                self.version = len(list(model_artifacts_dir.glob(f"v*"))) + 1
            self.instance_name = f"{self.MODEL_NAME}_v{self.version}"
            self.instance_dir = model_artifacts_dir / f"v{self.version}"
            self.instance_dir.mkdir(exist_ok=True)
        else:  # non-parametric models can just overwrite the same files since they don't have learnable parameters that change with each run
            self.instance_name = f"{self.MODEL_NAME}"
            self.instance_dir = model_artifacts_dir

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

    @abstractmethod
    def save(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, version: int = None) -> "TennisPredictorModel":
        pass
