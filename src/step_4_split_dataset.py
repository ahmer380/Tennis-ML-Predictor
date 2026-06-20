from typing import Tuple

import pandas as pd

FINALISED_ML_FEATURES = [
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
    "player_A_global_matches_played_last_365",
    "player_B_global_matches_played_last_365",
    "global_matches_played_last_365_diff",
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
    "player_A_surface_win_pct_last_100",
    "player_B_surface_win_pct_last_100",
    "surface_win_pct_last_100_diff",
    # Head-to-head
    "player_A_h2h_wins",
    "player_B_h2h_wins",
    "h2h_diff",
    # Game stats
    "player_A_ace_pct",
    "player_B_ace_pct",
    "ace_pct_diff",
    "player_A_df_pct",
    "player_B_df_pct",
    "df_pct_diff",
    "player_A_1st_in_pct",
    "player_B_1st_in_pct",
    "1st_in_pct_diff",
    "player_A_1st_won_pct",
    "player_B_1st_won_pct",
    "1st_won_pct_diff",
    "player_A_2nd_won_pct",
    "player_B_2nd_won_pct",
    "2nd_won_pct_diff",
    "player_A_bp_saved_pct",
    "player_B_bp_saved_pct",
    "bp_saved_pct_diff",
    "player_A_rp_won_pct",
    "player_B_rp_won_pct",
    "rp_won_pct_diff",
    "player_A_bp_won_pct",
    "player_B_bp_won_pct",
    "bp_won_pct_diff",
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


def split_dataset(
    df_features: pd.DataFrame, train_size: float = 0.7, validation_size: float = 0.15, test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and testing sets, preserving chronological order.
    """
    assert train_size + validation_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"

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
