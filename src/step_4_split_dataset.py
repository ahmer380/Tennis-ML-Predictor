from typing import Tuple

import pandas as pd


def split_dataset(
    df_features: pd.DataFrame, train_size: float = 0.7, validation_size: float = 0.15, test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and testing sets, preserving chronological order.
    """
    assert train_size + validation_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"

    finalised_ml_features = [
        "rank_A",
        "rank_B",
        "rank_diff",
        "global_elo_A",
        "global_elo_B",
        "global_elo_diff",
        "surface_elo_A",
        "surface_elo_B",
        "surface_elo_diff",
        "h2h_wins_A",
        "h2h_wins_B",
        "h2h_diff",
        "age_A",
        "age_B",
        "age_diff",
        "tournament_minutes_A",
        "tournament_minutes_B",
        "tournament_minutes_diff",
        "hard_surface",
        "clay_surface",
        "grass_surface",
        "best_of_5",
        "player_A_win",
    ]

    dfc = df_features.copy()
    dfc = dfc[finalised_ml_features]

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
