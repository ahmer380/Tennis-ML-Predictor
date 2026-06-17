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
        # Ranking
        "player_A_rank",
        "player_B_rank",
        "rank_diff",
        "player_A_rank_points",
        "player_B_rank_points",
        "rank_points_diff",
        # Experience
        "player_A_global_matches_played",
        "player_B_global_matches_played",
        "global_matches_played_diff",
        "player_A_surface_matches_played",
        "player_B_surface_matches_played",
        "surface_matches_played_diff",
        # Elo
        "player_A_global_elo",
        "player_B_global_elo",
        "global_elo_diff",
        "player_A_surface_elo",
        "player_B_surface_elo",
        "surface_elo_diff",
        # Form
        "player_A_global_win_pct_last_10",
        "player_B_global_win_pct_last_10",
        "global_win_pct_last_10_diff",
        "player_A_global_win_pct_last_25",
        "player_B_global_win_pct_last_25",
        "global_win_pct_last_25_diff",
        "player_A_global_win_pct_last_50",
        "player_B_global_win_pct_last_50",
        "global_win_pct_last_50_diff",
        "player_A_global_win_pct_last_100",
        "player_B_global_win_pct_last_100",
        "global_win_pct_last_100_diff",
        "player_A_surface_win_pct_last_10",
        "player_B_surface_win_pct_last_10",
        "surface_win_pct_last_10_diff",
        # Head-to-head
        "player_A_h2h_wins",
        "player_B_h2h_wins",
        "h2h_diff",
        # Physical
        "player_A_age",
        "player_B_age",
        "age_diff",
        "player_A_ht",
        "player_B_ht",
        "ht_diff",
        # Tournament fatigue
        "player_A_tournament_minutes",
        "player_B_tournament_minutes",
        "tournament_minutes_diff",
        # Match context
        "best_of_5",
        "hard_surface",
        "clay_surface",
        "grass_surface",
        # Target
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
