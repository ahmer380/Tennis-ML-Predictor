import argparse

import pandas as pd

from src.data.download_dataset import download_dataset
from src.data.load_dataset import load_dataset
from src.data.preprocess_dataset import preprocess_dataset

from src.feature.feature_engineering import FeatureEngineer
from src.feature.player_profile import PlayerProfile
from src.feature.features import FINALISED_ML_FEATURES

from src.models.model import TennisPredictorModel
from src.models.elo import TennisPredictorElo
from src.models.mlp import TennisPredictorMLP
from src.models.xgboost import TennisPredictorXGBoost


def predict(
    model_type: str,
    player_a_name: str,
    player_b_name: str,
    surface: str,
    best_of: int,
    player_a_year: int = pd.Timestamp.now().year,
    player_b_year: int = pd.Timestamp.now().year,
) -> float:
    """Pipeline to predict the win probability of Player A winning against Player B using the specified model type."""

    player_a_profile = build_player_profile(player_a_name, player_a_year)
    player_b_profile = build_player_profile(player_b_name, player_b_year)

    if model_type == "elo":
        model = TennisPredictorElo()
    elif model_type == "mlp":
        model = TennisPredictorMLP.load(version=1)
    elif model_type == "xgboost":
        model = TennisPredictorXGBoost.load(version=1)

    player_a_win_probability = (
        _predict_direction(model, player_a_profile, player_b_profile, surface, best_of)
        + (1 - _predict_direction(model, player_b_profile, player_a_profile, surface, best_of))
    ) / 2  # Average the predictions across both directions (A vs B and B vs A) to account for potential asymmetry in the model

    return player_a_win_probability


def build_player_profile(player_name: str, player_year: int) -> PlayerProfile:
    """Build a player profile for a given player up to a specified year."""
    download_dataset()
    df = load_dataset()
    df_player_year = df[
        (pd.to_datetime(df["tourney_date"], format="%Y%m%d") <= pd.Timestamp(year=player_year, month=12, day=31))
        & ((df["winner_name"] == player_name) | (df["loser_name"] == player_name))
    ]
    df_preprocessed = preprocess_dataset(df_player_year)
    feature_engineer = FeatureEngineer()
    player_profiles = feature_engineer.build_player_profiles(df_preprocessed)

    player_profile = next((profile for profile in player_profiles.values() if profile.name == player_name), None)
    if not player_profile:
        raise ValueError(f"Player profile for {player_name} up to {player_year} not found in the dataset.")

    return player_profile


def _predict_direction(
    model: TennisPredictorModel,
    player_a_profile: PlayerProfile,
    player_b_profile: PlayerProfile,
    surface: str,
    best_of: int,
) -> float:
    """predict the probability of Player A winning against Player B in a single direction (A vs B)."""

    feature_vector = {
        # Ranking
        "player_A_rank": player_a_profile.rank,
        "player_B_rank": player_b_profile.rank,
        "rank_diff": player_a_profile.rank - player_b_profile.rank,
        "player_A_rank_points": player_a_profile.rank_points,
        "player_B_rank_points": player_b_profile.rank_points,
        "rank_points_diff": player_a_profile.rank_points - player_b_profile.rank_points,
        # Experience
        "player_A_global_matches_played": player_a_profile.get_matches_played("global"),
        "player_B_global_matches_played": player_b_profile.get_matches_played("global"),
        "global_matches_played_diff": player_a_profile.get_matches_played("global")
        - player_b_profile.get_matches_played("global"),
        "player_A_surface_matches_played": player_a_profile.get_matches_played(surface),
        "player_B_surface_matches_played": player_b_profile.get_matches_played(surface),
        "surface_matches_played_diff": player_a_profile.get_matches_played(surface)
        - player_b_profile.get_matches_played(surface),
        "player_A_global_matches_played_last_365": player_a_profile.get_matches_played(
            "global", pd.Timestamp.now() - pd.DateOffset(years=1)
        ),
        "player_B_global_matches_played_last_365": player_b_profile.get_matches_played(
            "global", pd.Timestamp.now() - pd.DateOffset(years=1)
        ),
        "global_matches_played_last_365_diff": player_a_profile.get_matches_played(
            "global", pd.Timestamp.now() - pd.DateOffset(years=1)
        )
        - player_b_profile.get_matches_played("global", pd.Timestamp.now() - pd.DateOffset(years=1)),
        # Elo
        "player_A_global_elo": player_a_profile.elos["global"],
        "player_B_global_elo": player_b_profile.elos["global"],
        "global_elo_diff": player_a_profile.elos["global"] - player_b_profile.elos["global"],
        "player_A_surface_elo": player_a_profile.elos[surface],
        "player_B_surface_elo": player_b_profile.elos[surface],
        "surface_elo_diff": player_a_profile.elos[surface] - player_b_profile.elos[surface],
        # Form
        "player_A_global_win_pct_last_10": player_a_profile.get_recent_win_percentage(10, "global"),
        "player_B_global_win_pct_last_10": player_b_profile.get_recent_win_percentage(10, "global"),
        "global_win_pct_last_10_diff": player_a_profile.get_recent_win_percentage(10, "global")
        - player_b_profile.get_recent_win_percentage(10, "global"),
        "player_A_global_win_pct_last_25": player_a_profile.get_recent_win_percentage(25, "global"),
        "player_B_global_win_pct_last_25": player_b_profile.get_recent_win_percentage(25, "global"),
        "global_win_pct_last_25_diff": player_a_profile.get_recent_win_percentage(25, "global")
        - player_b_profile.get_recent_win_percentage(25, "global"),
        "player_A_global_win_pct_last_50": player_a_profile.get_recent_win_percentage(50, "global"),
        "player_B_global_win_pct_last_50": player_b_profile.get_recent_win_percentage(50, "global"),
        "global_win_pct_last_50_diff": player_a_profile.get_recent_win_percentage(50, "global")
        - player_b_profile.get_recent_win_percentage(50, "global"),
        "player_A_global_win_pct_last_100": player_a_profile.get_recent_win_percentage(100, "global"),
        "player_B_global_win_pct_last_100": player_b_profile.get_recent_win_percentage(100, "global"),
        "global_win_pct_last_100_diff": player_a_profile.get_recent_win_percentage(100, "global")
        - player_b_profile.get_recent_win_percentage(100, "global"),
        "player_A_surface_win_pct_last_100": player_a_profile.get_recent_win_percentage(100, surface),
        "player_B_surface_win_pct_last_100": player_b_profile.get_recent_win_percentage(100, surface),
        "surface_win_pct_last_100_diff": player_a_profile.get_recent_win_percentage(100, surface)
        - player_b_profile.get_recent_win_percentage(100, surface),
        # Head-to-head
        "player_A_h2h_wins": player_a_profile.get_h2h_wins(player_b_profile.id),
        "player_B_h2h_wins": player_b_profile.get_h2h_wins(player_a_profile.id),
        "h2h_diff": player_a_profile.get_h2h_wins(player_b_profile.id)
        - player_b_profile.get_h2h_wins(player_a_profile.id),
        # Game stats
        "player_A_ace_pct": player_a_profile.get_recent_game_stat_average("p_ace", 100, 0.08),
        "player_B_ace_pct": player_b_profile.get_recent_game_stat_average("p_ace", 100, 0.08),
        "ace_pct_diff": player_a_profile.get_recent_game_stat_average("p_ace", 100, 0.08)
        - player_b_profile.get_recent_game_stat_average("p_ace", 100, 0.08),
        "player_A_df_pct": player_a_profile.get_recent_game_stat_average("p_df", 100, 0.05),
        "player_B_df_pct": player_b_profile.get_recent_game_stat_average("p_df", 100, 0.05),
        "df_pct_diff": player_a_profile.get_recent_game_stat_average("p_df", 100, 0.05)
        - player_b_profile.get_recent_game_stat_average("p_df", 100, 0.05),
        "player_A_1st_in_pct": player_a_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5),
        "player_B_1st_in_pct": player_b_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5),
        "1st_in_pct_diff": player_a_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5),
        "player_A_1st_won_pct": player_a_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5),
        "player_B_1st_won_pct": player_b_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5),
        "1st_won_pct_diff": player_a_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5),
        "player_A_2nd_won_pct": player_a_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5),
        "player_B_2nd_won_pct": player_b_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5),
        "2nd_won_pct_diff": player_a_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5),
        "player_A_bp_saved_pct": player_a_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5),
        "player_B_bp_saved_pct": player_b_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5),
        "bp_saved_pct_diff": player_a_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5),
        "player_A_rp_won_pct": player_a_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5),
        "player_B_rp_won_pct": player_b_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5),
        "rp_won_pct_diff": player_a_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5),
        "player_A_bp_won_pct": player_a_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5),
        "player_B_bp_won_pct": player_b_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5),
        "bp_won_pct_diff": player_a_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5),
        # Physical
        "player_A_age": player_a_profile.age,
        "player_B_age": player_b_profile.age,
        "age_diff": player_a_profile.age - player_b_profile.age,
        "player_A_ht": player_a_profile.ht,
        "player_B_ht": player_b_profile.ht,
        "ht_diff": player_a_profile.ht - player_b_profile.ht,
        # Tournament fatigue
        "player_A_tournament_minutes": 0,
        "player_B_tournament_minutes": 0,
        "tournament_minutes_diff": 0,
        # Match context
        "best_of_5": 1 if best_of == 5 else 0,
        "hard_surface": 1 if surface == "Hard" else 0,
        "clay_surface": 1 if surface == "Clay" else 0,
        "grass_surface": 1 if surface == "Grass" else 0,
    }

    feature_vector = {k: v for k, v in feature_vector.items() if k in FINALISED_ML_FEATURES}
    if set(feature_vector.keys()) != set(FINALISED_ML_FEATURES) - {"player_A_win"}:
        raise ValueError("Feature vector does not contain all required features for the model.")

    win_probability = model.predict(pd.DataFrame([feature_vector]))[0]
    return win_probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the outcome of a tennis match between two players.")
    parser.add_argument(
        "--player_a_name",
        type=str,
        required=True,
        help="Name of the first player",
    )
    parser.add_argument(
        "--player_b_name",
        type=str,
        required=True,
        help="Name of the second player",
    )
    parser.add_argument(
        "--surface",
        type=str,
        default="Hard",
        choices=["Hard", "Clay", "Grass"],
        help="The surface of the court",
    )
    parser.add_argument(
        "--best_of",
        type=int,
        default=3,
        choices=[3, 5],
        help="The best-of value for the match",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["elo", "mlp", "xgboost"],
        help="The model type to use for prediction",
    )
    parser.add_argument(
        "--player_a_year",
        type=int,
        default=pd.Timestamp.now().year,
        help="The end year of the first player to consider",
    )
    parser.add_argument(
        "--player_b_year",
        type=int,
        default=pd.Timestamp.now().year,
        help="The end year of the second player to consider",
    )
    args = parser.parse_args()

    player_a_win_probability = predict(
        model_type=args.model,
        player_a_name=args.player_a_name,
        player_b_name=args.player_b_name,
        surface=args.surface,
        best_of=args.best_of,
        player_a_year=args.player_a_year,
        player_b_year=args.player_b_year,
    )

    print("=" * 50)
    print(f"Win probability according to {args.model}:")
    print(f"{args.player_a_name}: {player_a_win_probability:.2%}")
    print(f"{args.player_b_name}: {1 - player_a_win_probability:.2%}")
    print("=" * 50)
    print()
