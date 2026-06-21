from typing import Dict, Tuple

import pandas as pd

from src.feature.elo_rating_engine import EloRatingEngine
from src.feature.player_profile import PlayerProfile

elo_engine = EloRatingEngine()


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, PlayerProfile]]:
    """Chronological pass that produces pre-match features using only past information."""
    dfc = df.copy()

    # Sort matches chronologically (oldest to newest) to prevent future data leakage in feature engineering
    dfc = dfc.sort_values(by=["tourney_date", "match_num"], kind="mergesort").reset_index(drop=True)

    # key=player_id, value=PlayerProfile object
    player_profiles: Dict[int, PlayerProfile] = {}
    elo_engine = EloRatingEngine()

    # List of extended features to collect in order
    features = {
        "rank_diff": [],
        "rank_points_diff": [],
        "age_diff": [],
        "ht_diff": [],
        "player_A_global_matches_played": [],
        "player_B_global_matches_played": [],
        "global_matches_played_diff": [],
        "player_A_surface_matches_played": [],
        "player_B_surface_matches_played": [],
        "surface_matches_played_diff": [],
        "player_A_global_matches_played_last_365": [],
        "player_B_global_matches_played_last_365": [],
        "global_matches_played_last_365_diff": [],
        "player_A_global_win_pct_last_10": [],
        "player_B_global_win_pct_last_10": [],
        "global_win_pct_last_10_diff": [],
        "player_A_global_win_pct_last_25": [],
        "player_B_global_win_pct_last_25": [],
        "global_win_pct_last_25_diff": [],
        "player_A_global_win_pct_last_50": [],
        "player_B_global_win_pct_last_50": [],
        "global_win_pct_last_50_diff": [],
        "player_A_global_win_pct_last_100": [],
        "player_B_global_win_pct_last_100": [],
        "global_win_pct_last_100_diff": [],
        "player_A_surface_win_pct_last_100": [],
        "player_B_surface_win_pct_last_100": [],
        "surface_win_pct_last_100_diff": [],
        "player_A_global_elo": [],
        "player_B_global_elo": [],
        "global_elo_diff": [],
        "player_A_surface_elo": [],
        "player_B_surface_elo": [],
        "surface_elo_diff": [],
        "player_A_h2h_wins": [],
        "player_B_h2h_wins": [],
        "h2h_diff": [],
        "player_A_ace_pct": [],
        "player_B_ace_pct": [],
        "ace_pct_diff": [],
        "player_A_df_pct": [],
        "player_B_df_pct": [],
        "df_pct_diff": [],
        "player_A_1st_in_pct": [],
        "player_B_1st_in_pct": [],
        "1st_in_pct_diff": [],
        "player_A_1st_won_pct": [],
        "player_B_1st_won_pct": [],
        "1st_won_pct_diff": [],
        "player_A_2nd_won_pct": [],
        "player_B_2nd_won_pct": [],
        "2nd_won_pct_diff": [],
        "player_A_bp_saved_pct": [],
        "player_B_bp_saved_pct": [],
        "bp_saved_pct_diff": [],
        "player_A_rp_won_pct": [],
        "player_B_rp_won_pct": [],
        "rp_won_pct_diff": [],
        "player_A_bp_won_pct": [],
        "player_B_bp_won_pct": [],
        "bp_won_pct_diff": [],
        "player_A_tournament_minutes": [],
        "player_B_tournament_minutes": [],
        "tournament_minutes_diff": [],
        "hard_surface": [],
        "clay_surface": [],
        "grass_surface": [],
        "best_of_5": [],
    }

    # Iterate chronologically and build pre-match snapshots
    for _, row in dfc.iterrows():
        if row["player_A_id"] not in player_profiles:
            player_profiles[row["player_A_id"]] = PlayerProfile(id=row["player_A_id"], name=row["player_A_name"])
        if row["player_B_id"] not in player_profiles:
            player_profiles[row["player_B_id"]] = PlayerProfile(id=row["player_B_id"], name=row["player_B_name"])

        player_a_profile = player_profiles[row["player_A_id"]]
        player_b_profile = player_profiles[row["player_B_id"]]

        # Pre-match updates to player profiles (Elo updates occur externally)
        player_a_profile.pre_match_update(row)
        player_b_profile.pre_match_update(row)

        # Compute feature differences
        features["rank_diff"].append(row["player_A_rank"] - row["player_B_rank"])
        features["rank_points_diff"].append(row["player_A_rank_points"] - row["player_B_rank_points"])
        features["age_diff"].append(row["player_A_age"] - row["player_B_age"])
        features["ht_diff"].append(row["player_A_ht"] - row["player_B_ht"])

        # Compute experience features
        features["player_A_global_matches_played"].append(player_a_profile.get_matches_played("global"))
        features["player_B_global_matches_played"].append(player_b_profile.get_matches_played("global"))
        features["global_matches_played_diff"].append(
            player_a_profile.get_matches_played("global") - player_b_profile.get_matches_played("global")
        )

        features["player_A_surface_matches_played"].append(player_a_profile.get_matches_played(row["surface"]))
        features["player_B_surface_matches_played"].append(player_b_profile.get_matches_played(row["surface"]))
        features["surface_matches_played_diff"].append(
            player_a_profile.get_matches_played(row["surface"]) - player_b_profile.get_matches_played(row["surface"])
        )

        one_year_ago = pd.to_datetime(row["tourney_date"], format="%Y%m%d") - pd.DateOffset(years=1)
        player_a_global_matches_played_last_365 = player_a_profile.get_matches_played("global", from_date=one_year_ago)
        player_b_global_matches_played_last_365 = player_b_profile.get_matches_played("global", from_date=one_year_ago)
        features["player_A_global_matches_played_last_365"].append(player_a_global_matches_played_last_365)
        features["player_B_global_matches_played_last_365"].append(player_b_global_matches_played_last_365)
        features["global_matches_played_last_365_diff"].append(
            player_a_global_matches_played_last_365 - player_b_global_matches_played_last_365
        )

        # Compute form features
        player_a_global_win_pct_last_10 = player_a_profile.get_recent_win_percentage(10, "global")
        player_b_global_win_pct_last_10 = player_b_profile.get_recent_win_percentage(10, "global")
        features["player_A_global_win_pct_last_10"].append(player_a_global_win_pct_last_10)
        features["player_B_global_win_pct_last_10"].append(player_b_global_win_pct_last_10)
        features["global_win_pct_last_10_diff"].append(
            player_a_global_win_pct_last_10 - player_b_global_win_pct_last_10
        )

        player_a_global_win_pct_last_25 = player_a_profile.get_recent_win_percentage(25, "global")
        player_b_global_win_pct_last_25 = player_b_profile.get_recent_win_percentage(25, "global")
        features["player_A_global_win_pct_last_25"].append(player_a_global_win_pct_last_25)
        features["player_B_global_win_pct_last_25"].append(player_b_global_win_pct_last_25)
        features["global_win_pct_last_25_diff"].append(
            player_a_global_win_pct_last_25 - player_b_global_win_pct_last_25
        )

        player_a_global_win_pct_last_50 = player_a_profile.get_recent_win_percentage(50, "global")
        player_b_global_win_pct_last_50 = player_b_profile.get_recent_win_percentage(50, "global")
        features["player_A_global_win_pct_last_50"].append(player_a_global_win_pct_last_50)
        features["player_B_global_win_pct_last_50"].append(player_b_global_win_pct_last_50)
        features["global_win_pct_last_50_diff"].append(
            player_a_global_win_pct_last_50 - player_b_global_win_pct_last_50
        )

        player_a_global_win_pct_last_100 = player_a_profile.get_recent_win_percentage(100, "global")
        player_b_global_win_pct_last_100 = player_b_profile.get_recent_win_percentage(100, "global")
        features["player_A_global_win_pct_last_100"].append(player_a_global_win_pct_last_100)
        features["player_B_global_win_pct_last_100"].append(player_b_global_win_pct_last_100)
        features["global_win_pct_last_100_diff"].append(
            player_a_global_win_pct_last_100 - player_b_global_win_pct_last_100
        )

        player_a_surface_win_pct_last_100 = player_a_profile.get_recent_win_percentage(100, row["surface"])
        player_b_surface_win_pct_last_100 = player_b_profile.get_recent_win_percentage(100, row["surface"])
        features["player_A_surface_win_pct_last_100"].append(player_a_surface_win_pct_last_100)
        features["player_B_surface_win_pct_last_100"].append(player_b_surface_win_pct_last_100)
        features["surface_win_pct_last_100_diff"].append(
            player_a_surface_win_pct_last_100 - player_b_surface_win_pct_last_100
        )

        # Compute Elo features
        features["player_A_global_elo"].append(player_a_profile.elos["global"])
        features["player_B_global_elo"].append(player_b_profile.elos["global"])
        features["global_elo_diff"].append(player_a_profile.elos["global"] - player_b_profile.elos["global"])

        features["player_A_surface_elo"].append(player_a_profile.elos[row["surface"]])
        features["player_B_surface_elo"].append(player_b_profile.elos[row["surface"]])
        features["surface_elo_diff"].append(
            player_a_profile.elos[row["surface"]] - player_b_profile.elos[row["surface"]]
        )

        # Compute head-to-head features
        h2h_wins_a = player_a_profile.get_h2h_wins(row["player_B_id"])
        h2h_wins_b = player_b_profile.get_h2h_wins(row["player_A_id"])
        features["player_A_h2h_wins"].append(h2h_wins_a)
        features["player_B_h2h_wins"].append(h2h_wins_b)
        features["h2h_diff"].append(h2h_wins_a - h2h_wins_b)

        # Compute player game stats features
        player_a_ace_pct = player_a_profile.get_recent_game_stat_average("p_ace", 100, 0.08)
        player_b_ace_pct = player_b_profile.get_recent_game_stat_average("p_ace", 100, 0.08)
        features["player_A_ace_pct"].append(player_a_ace_pct)
        features["player_B_ace_pct"].append(player_b_ace_pct)
        features["ace_pct_diff"].append(player_a_ace_pct - player_b_ace_pct)

        player_a_df_pct = player_a_profile.get_recent_game_stat_average("p_df", 100, 0.05)
        player_b_df_pct = player_b_profile.get_recent_game_stat_average("p_df", 100, 0.05)
        features["player_A_df_pct"].append(player_a_df_pct)
        features["player_B_df_pct"].append(player_b_df_pct)
        features["df_pct_diff"].append(player_a_df_pct - player_b_df_pct)

        player_a_1st_in_pct = player_a_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5)
        player_b_1st_in_pct = player_b_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5)
        features["player_A_1st_in_pct"].append(player_a_1st_in_pct)
        features["player_B_1st_in_pct"].append(player_b_1st_in_pct)
        features["1st_in_pct_diff"].append(player_a_1st_in_pct - player_b_1st_in_pct)

        player_a_1st_won_pct = player_a_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5)
        player_b_1st_won_pct = player_b_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5)
        features["player_A_1st_won_pct"].append(player_a_1st_won_pct)
        features["player_B_1st_won_pct"].append(player_b_1st_won_pct)
        features["1st_won_pct_diff"].append(player_a_1st_won_pct - player_b_1st_won_pct)

        player_a_2nd_won_pct = player_a_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5)
        player_b_2nd_won_pct = player_b_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5)
        features["player_A_2nd_won_pct"].append(player_a_2nd_won_pct)
        features["player_B_2nd_won_pct"].append(player_b_2nd_won_pct)
        features["2nd_won_pct_diff"].append(player_a_2nd_won_pct - player_b_2nd_won_pct)

        player_a_bp_saved_pct = player_a_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5)
        player_b_bp_saved_pct = player_b_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5)
        features["player_A_bp_saved_pct"].append(player_a_bp_saved_pct)
        features["player_B_bp_saved_pct"].append(player_b_bp_saved_pct)
        features["bp_saved_pct_diff"].append(player_a_bp_saved_pct - player_b_bp_saved_pct)

        player_a_rp_won_pct = player_a_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5)
        player_b_rp_won_pct = player_b_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5)
        features["player_A_rp_won_pct"].append(player_a_rp_won_pct)
        features["player_B_rp_won_pct"].append(player_b_rp_won_pct)
        features["rp_won_pct_diff"].append(player_a_rp_won_pct - player_b_rp_won_pct)

        player_a_bp_won_pct = player_a_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5)
        player_b_bp_won_pct = player_b_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5)
        features["player_A_bp_won_pct"].append(player_a_bp_won_pct)
        features["player_B_bp_won_pct"].append(player_b_bp_won_pct)
        features["bp_won_pct_diff"].append(player_a_bp_won_pct - player_b_bp_won_pct)

        # Compute tournament fatigue features
        features["player_A_tournament_minutes"].append(player_a_profile.current_tournament_minutes_played)
        features["player_B_tournament_minutes"].append(player_b_profile.current_tournament_minutes_played)
        features["tournament_minutes_diff"].append(
            player_a_profile.current_tournament_minutes_played - player_b_profile.current_tournament_minutes_played
        )

        # Compute match context features
        features["hard_surface"].append(1 if row["surface"] == "Hard" else 0)
        features["clay_surface"].append(1 if row["surface"] == "Clay" else 0)
        features["grass_surface"].append(1 if row["surface"] == "Grass" else 0)

        features["best_of_5"].append(1 if row["best_of"] == 5 else 0)

        # Post-match updates to player profiles (Elo updates occur externally)
        player_a_profile.post_match_update(row, "player_A")
        player_b_profile.post_match_update(row, "player_B")

        player_a_profile.elos["global"], player_b_profile.elos["global"] = elo_engine.update_ratings(
            elo_a=player_a_profile.elos["global"],
            elo_b=player_b_profile.elos["global"],
            matches_played_a=player_a_profile.get_matches_played("global"),
            matches_played_b=player_b_profile.get_matches_played("global"),
            tourney_level=row["tourney_level"],
            score_a=row["player_A_win"],
        )
        player_a_profile.elos[row["surface"]], player_b_profile.elos[row["surface"]] = elo_engine.update_ratings(
            elo_a=player_a_profile.elos[row["surface"]],
            elo_b=player_b_profile.elos[row["surface"]],
            matches_played_a=player_a_profile.get_matches_played(row["surface"]),
            matches_played_b=player_b_profile.get_matches_played(row["surface"]),
            tourney_level=row["tourney_level"],
            score_a=row["player_A_win"],
        )

    return (dfc.assign(**features), player_profiles)


def get_player_profile_by_name(player_profiles: Dict[int, PlayerProfile], player_name: str) -> PlayerProfile:
    """Retrieve a PlayerProfile object by player name."""
    for profile in player_profiles.values():
        if profile.name == player_name:
            return profile

    raise ValueError(f"Player '{player_name}' not found in player profiles.")
