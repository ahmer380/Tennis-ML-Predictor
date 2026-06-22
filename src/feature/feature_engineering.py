from typing import Dict

import pandas as pd

from src.feature.elo_rating_engine import EloRatingEngine
from src.feature.player_profile import PlayerProfile


class FeatureEngineer:
    """Class to handle feature engineering for tennis match data."""

    def __init__(self):
        self.elo_engine = EloRatingEngine()
        self.player_profiles: Dict[int, PlayerProfile] = {}

    def build_player_profiles(self, df_preprocessed: pd.DataFrame) -> Dict[int, PlayerProfile]:
        """Build player profiles from the preprocessed DataFrame of ATP matches."""
        print("\nBuilding player profiles...")

        dfc = df_preprocessed.copy()
        # Sort matches chronologically (oldest to newest) to prevent future data leakage in feature engineering
        dfc = dfc.sort_values(by=["tourney_date", "match_num"], kind="mergesort").reset_index(drop=True)

        for _, row in dfc.iterrows():
            player_a = self._get_player_profile(row["player_A_id"], row["player_A_name"])
            player_b = self._get_player_profile(row["player_B_id"], row["player_B_name"])

            player_a.pre_match_update(row)
            player_b.pre_match_update(row)

            player_a.post_match_update(row, "player_A")
            player_b.post_match_update(row, "player_B")
            player_a.elos["global"], player_b.elos["global"] = self.elo_engine.update_ratings(
                elo_a=player_a.elos["global"],
                elo_b=player_b.elos["global"],
                matches_played_a=player_a.get_matches_played("global"),
                matches_played_b=player_b.get_matches_played("global"),
                tourney_level=row["tourney_level"],
                score_a=row["player_A_win"],
            )
            player_a.elos[row["surface"]], player_b.elos[row["surface"]] = self.elo_engine.update_ratings(
                elo_a=player_a.elos[row["surface"]],
                elo_b=player_b.elos[row["surface"]],
                matches_played_a=player_a.get_matches_played(row["surface"]),
                matches_played_b=player_b.get_matches_played(row["surface"]),
                tourney_level=row["tourney_level"],
                score_a=row["player_A_win"],
            )

        return self.player_profiles

    def engineer_dataframe(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        """Add Engineer features to the preprocessed DataFrame of ATP matches."""
        print("\nEngineering features...")

        features = (
            []
        )  # Same as before, but we also compute features for each match and store them in a list of dictionaries.

        dfc = df_preprocessed.copy()
        # Sort matches chronologically (oldest to newest) to prevent future data leakage in feature engineering
        dfc = dfc.sort_values(by=["tourney_date", "match_num"], kind="mergesort").reset_index(drop=True)

        for _, row in dfc.iterrows():
            player_a = self._get_player_profile(row["player_A_id"], row["player_A_name"])
            player_b = self._get_player_profile(row["player_B_id"], row["player_B_name"])

            player_a.pre_match_update(row)
            player_b.pre_match_update(row)

            features.append(self._compute_features(row, player_a, player_b))

            player_a.post_match_update(row, "player_A")
            player_b.post_match_update(row, "player_B")
            player_a.elos["global"], player_b.elos["global"] = self.elo_engine.update_ratings(
                elo_a=player_a.elos["global"],
                elo_b=player_b.elos["global"],
                matches_played_a=player_a.get_matches_played("global"),
                matches_played_b=player_b.get_matches_played("global"),
                tourney_level=row["tourney_level"],
                score_a=row["player_A_win"],
            )
            player_a.elos[row["surface"]], player_b.elos[row["surface"]] = self.elo_engine.update_ratings(
                elo_a=player_a.elos[row["surface"]],
                elo_b=player_b.elos[row["surface"]],
                matches_played_a=player_a.get_matches_played(row["surface"]),
                matches_played_b=player_b.get_matches_played(row["surface"]),
                tourney_level=row["tourney_level"],
                score_a=row["player_A_win"],
            )

        return dfc.assign(**pd.DataFrame(features, index=dfc.index))

    def _get_player_profile(self, player_id: int, player_name: str) -> PlayerProfile:
        """Retrieve or create a PlayerProfile object for a given player ID."""
        if player_id not in self.player_profiles:
            self.player_profiles[player_id] = PlayerProfile(id=player_id, name=player_name)
        return self.player_profiles[player_id]

    def _compute_features(self, row: pd.Series, player_a: PlayerProfile, player_b: PlayerProfile) -> Dict:
        """Calculate features for a given match row and player profiles."""
        features = {}

        # Compute existing feature differences
        features["rank_diff"] = row["player_A_rank"] - row["player_B_rank"]
        features["rank_points_diff"] = row["player_A_rank_points"] - row["player_B_rank_points"]
        features["age_diff"] = row["player_A_age"] - row["player_B_age"]
        features["ht_diff"] = row["player_A_ht"] - row["player_B_ht"]

        # Compute experience features
        features["player_A_global_matches_played"] = player_a.get_matches_played("global")
        features["player_B_global_matches_played"] = player_b.get_matches_played("global")
        features["global_matches_played_diff"] = player_a.get_matches_played("global") - player_b.get_matches_played(
            "global"
        )

        features["player_A_surface_matches_played"] = player_a.get_matches_played(row["surface"])
        features["player_B_surface_matches_played"] = player_b.get_matches_played(row["surface"])
        features["surface_matches_played_diff"] = player_a.get_matches_played(
            row["surface"]
        ) - player_b.get_matches_played(row["surface"])

        one_year_ago = pd.to_datetime(row["tourney_date"], format="%Y%m%d") - pd.DateOffset(years=1)
        player_a_global_matches_played_last_365 = player_a.get_matches_played("global", from_date=one_year_ago)
        player_b_global_matches_played_last_365 = player_b.get_matches_played("global", from_date=one_year_ago)
        features["player_A_global_matches_played_last_365"] = player_a_global_matches_played_last_365
        features["player_B_global_matches_played_last_365"] = player_b_global_matches_played_last_365
        features["global_matches_played_last_365_diff"] = (
            player_a_global_matches_played_last_365 - player_b_global_matches_played_last_365
        )

        # Compute form features
        player_a_global_win_pct_last_10 = player_a.get_recent_win_percentage(10, "global")
        player_b_global_win_pct_last_10 = player_b.get_recent_win_percentage(10, "global")
        features["player_A_global_win_pct_last_10"] = player_a_global_win_pct_last_10
        features["player_B_global_win_pct_last_10"] = player_b_global_win_pct_last_10
        features["global_win_pct_last_10_diff"] = player_a_global_win_pct_last_10 - player_b_global_win_pct_last_10

        player_a_global_win_pct_last_25 = player_a.get_recent_win_percentage(25, "global")
        player_b_global_win_pct_last_25 = player_b.get_recent_win_percentage(25, "global")
        features["player_A_global_win_pct_last_25"] = player_a_global_win_pct_last_25
        features["player_B_global_win_pct_last_25"] = player_b_global_win_pct_last_25
        features["global_win_pct_last_25_diff"] = player_a_global_win_pct_last_25 - player_b_global_win_pct_last_25

        player_a_global_win_pct_last_50 = player_a.get_recent_win_percentage(50, "global")
        player_b_global_win_pct_last_50 = player_b.get_recent_win_percentage(50, "global")
        features["player_A_global_win_pct_last_50"] = player_a_global_win_pct_last_50
        features["player_B_global_win_pct_last_50"] = player_b_global_win_pct_last_50
        features["global_win_pct_last_50_diff"] = player_a_global_win_pct_last_50 - player_b_global_win_pct_last_50

        player_a_global_win_pct_last_100 = player_a.get_recent_win_percentage(100, "global")
        player_b_global_win_pct_last_100 = player_b.get_recent_win_percentage(100, "global")
        features["player_A_global_win_pct_last_100"] = player_a_global_win_pct_last_100
        features["player_B_global_win_pct_last_100"] = player_b_global_win_pct_last_100
        features["global_win_pct_last_100_diff"] = player_a_global_win_pct_last_100 - player_b_global_win_pct_last_100

        player_a_surface_win_pct_last_100 = player_a.get_recent_win_percentage(100, row["surface"])
        player_b_surface_win_pct_last_100 = player_b.get_recent_win_percentage(100, row["surface"])
        features["player_A_surface_win_pct_last_100"] = player_a_surface_win_pct_last_100
        features["player_B_surface_win_pct_last_100"] = player_b_surface_win_pct_last_100
        features["surface_win_pct_last_100_diff"] = (
            player_a_surface_win_pct_last_100 - player_b_surface_win_pct_last_100
        )

        # Compute Elo features
        features["player_A_global_elo"] = player_a.elos["global"]
        features["player_B_global_elo"] = player_b.elos["global"]
        features["global_elo_diff"] = player_a.elos["global"] - player_b.elos["global"]

        features["player_A_surface_elo"] = player_a.elos[row["surface"]]
        features["player_B_surface_elo"] = player_b.elos[row["surface"]]
        features["surface_elo_diff"] = player_a.elos[row["surface"]] - player_b.elos[row["surface"]]

        # Compute head-to-head features
        h2h_wins_a = player_a.get_h2h_wins(row["player_B_id"])
        h2h_wins_b = player_b.get_h2h_wins(row["player_A_id"])
        features["player_A_h2h_wins"] = h2h_wins_a
        features["player_B_h2h_wins"] = h2h_wins_b
        features["h2h_diff"] = h2h_wins_a - h2h_wins_b

        # Compute player game stats features
        player_a_ace_pct = player_a.get_recent_game_stat_average("p_ace", 100, 0.08)
        player_b_ace_pct = player_b.get_recent_game_stat_average("p_ace", 100, 0.08)
        features["player_A_ace_pct"] = player_a_ace_pct
        features["player_B_ace_pct"] = player_b_ace_pct
        features["ace_pct_diff"] = player_a_ace_pct - player_b_ace_pct

        player_a_df_pct = player_a.get_recent_game_stat_average("p_df", 100, 0.05)
        player_b_df_pct = player_b.get_recent_game_stat_average("p_df", 100, 0.05)
        features["player_A_df_pct"] = player_a_df_pct
        features["player_B_df_pct"] = player_b_df_pct
        features["df_pct_diff"] = player_a_df_pct - player_b_df_pct

        player_a_1st_in_pct = player_a.get_recent_game_stat_average("p_1st_in", 100, 0.5)
        player_b_1st_in_pct = player_b.get_recent_game_stat_average("p_1st_in", 100, 0.5)
        features["player_A_1st_in_pct"] = player_a_1st_in_pct
        features["player_B_1st_in_pct"] = player_b_1st_in_pct
        features["1st_in_pct_diff"] = player_a_1st_in_pct - player_b_1st_in_pct

        player_a_1st_won_pct = player_a.get_recent_game_stat_average("p_1st_won", 100, 0.5)
        player_b_1st_won_pct = player_b.get_recent_game_stat_average("p_1st_won", 100, 0.5)
        features["player_A_1st_won_pct"] = player_a_1st_won_pct
        features["player_B_1st_won_pct"] = player_b_1st_won_pct
        features["1st_won_pct_diff"] = player_a_1st_won_pct - player_b_1st_won_pct

        player_a_2nd_won_pct = player_a.get_recent_game_stat_average("p_2nd_won", 100, 0.5)
        player_b_2nd_won_pct = player_b.get_recent_game_stat_average("p_2nd_won", 100, 0.5)
        features["player_A_2nd_won_pct"] = player_a_2nd_won_pct
        features["player_B_2nd_won_pct"] = player_b_2nd_won_pct
        features["2nd_won_pct_diff"] = player_a_2nd_won_pct - player_b_2nd_won_pct

        player_a_bp_saved_pct = player_a.get_recent_game_stat_average("p_bp_saved", 100, 0.5)
        player_b_bp_saved_pct = player_b.get_recent_game_stat_average("p_bp_saved", 100, 0.5)
        features["player_A_bp_saved_pct"] = player_a_bp_saved_pct
        features["player_B_bp_saved_pct"] = player_b_bp_saved_pct
        features["bp_saved_pct_diff"] = player_a_bp_saved_pct - player_b_bp_saved_pct

        player_a_rp_won_pct = player_a.get_recent_game_stat_average("p_rp_won", 100, 0.5)
        player_b_rp_won_pct = player_b.get_recent_game_stat_average("p_rp_won", 100, 0.5)
        features["player_A_rp_won_pct"] = player_a_rp_won_pct
        features["player_B_rp_won_pct"] = player_b_rp_won_pct
        features["rp_won_pct_diff"] = player_a_rp_won_pct - player_b_rp_won_pct

        player_a_bp_won_pct = player_a.get_recent_game_stat_average("p_bp_won", 100, 0.5)
        player_b_bp_won_pct = player_b.get_recent_game_stat_average("p_bp_won", 100, 0.5)
        features["player_A_bp_won_pct"] = player_a_bp_won_pct
        features["player_B_bp_won_pct"] = player_b_bp_won_pct
        features["bp_won_pct_diff"] = player_a_bp_won_pct - player_b_bp_won_pct

        # Compute tournament fatigue features
        features["player_A_tournament_minutes"] = player_a.current_tournament_minutes_played
        features["player_B_tournament_minutes"] = player_b.current_tournament_minutes_played
        features["tournament_minutes_diff"] = (
            player_a.current_tournament_minutes_played - player_b.current_tournament_minutes_played
        )

        # Compute match context features
        features["hard_surface"] = 1 if row["surface"] == "Hard" else 0
        features["clay_surface"] = 1 if row["surface"] == "Clay" else 0
        features["grass_surface"] = 1 if row["surface"] == "Grass" else 0

        features["best_of_5"] = 1 if row["best_of"] == 5 else 0

        return features


def get_player_profile_by_name(player_profiles: Dict[int, PlayerProfile], player_name: str) -> PlayerProfile:
    """Retrieve a PlayerProfile object by player name."""
    for profile in player_profiles.values():
        if profile.name == player_name:
            return profile

    raise ValueError(f"Player '{player_name}' not found in player profiles.")
