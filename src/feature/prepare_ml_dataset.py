from typing import Tuple

import pandas as pd

from src.feature.features import FINALISED_ML_FEATURES


def prepare_ml_dataset(
    df_features: pd.DataFrame, train_size: float = 0.7, validation_size: float = 0.15, test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare the dataset for machine learning by splitting it into training, validation, and test sets."""
    print(
        f"\nPreparing ML dataset with train size: {train_size}, validation size: {validation_size}, test size: {test_size}..."
    )

    assert train_size + validation_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"
    assert set(FINALISED_ML_FEATURES).issubset(df_features.columns), "Some features are missing from the dataset"

    dfc = df_features.copy()
    dfc = dfc[FINALISED_ML_FEATURES]

    train_end = int(len(dfc) * train_size)
    validation_end = train_end + int(len(dfc) * validation_size)

    train_df = dfc.iloc[:train_end]
    X_train = train_df.drop(columns=["player_A_win"])
    y_train = train_df["player_A_win"]

    validation_df = dfc.iloc[train_end:validation_end]
    X_validation = validation_df.drop(columns=["player_A_win"])
    y_validation = validation_df["player_A_win"]

    test_df = dfc.iloc[validation_end:]
    X_test = test_df.drop(columns=["player_A_win"])
    y_test = test_df["player_A_win"]

    return X_train, y_train, X_validation, y_validation, X_test, y_test
