from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

BASE_ELO = 1500.0


@dataclass
class _Match:
    """Class to hold match information"""

    date: pd.Timestamp
    won: bool


@dataclass
class PlayerProfile:
    """Class to hold player profile information, including Elo ratings, match history, and game stats."""

    # Bio and physical
    id: int
    name: str
    age: int = None
    ht: int = None

    # Ranking
    rank: int = None
    rank_points: int = None

    # Experience and form
    match_history: Dict[str, List[_Match]] = field(default_factory=lambda: defaultdict(list))

    # Elo
    elos: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: BASE_ELO))

    # Head-to-head
    h2h_records: Dict[int, List[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )  # key=opponent_id, value=win/loss record

    # Game stats
    p_ace: List[float] = field(default_factory=list)
    p_df: List[float] = field(default_factory=list)
    p_1st_in: List[float] = field(default_factory=list)
    p_1st_won: List[float] = field(default_factory=list)
    p_2nd_won: List[float] = field(default_factory=list)
    p_bp_saved: List[float] = field(default_factory=list)
    p_rp_won: List[float] = field(default_factory=list)
    p_bp_won: List[float] = field(default_factory=list)

    # Fatigue
    last_tournament_played_date: pd.Timestamp = None
    current_tournament_minutes_played: int = 0

    def get_matches_played(self, surface: str = "global", from_date: pd.Timestamp = None) -> int:
        """Return the number of matches played on a given surface from a specific date."""
        if not from_date:
            return len(self.match_history[surface])

        matches_played = 0
        for match in range(len(self.match_history[surface]) - 1, -1, -1):
            if self.match_history[surface][match].date >= from_date:
                matches_played += 1
            else:
                break

        return matches_played

    def get_h2h_wins(self, opponent_id: int) -> int:
        """Return the number of head-to-head wins against a specific opponent."""
        return self.h2h_records[opponent_id].count(True)

    def get_recent_win_percentage(self, num_matches: int, surface: str = "global") -> float:
        """Return the win percentage over the last `num_matches` on a given surface."""
        recent_matches = self.match_history[surface][-min(num_matches, len(self.match_history[surface])) :]
        return sum(match.won for match in recent_matches) / len(recent_matches) if recent_matches else 0.5

    def get_recent_game_stat_average(self, game_stat: str, num_matches: int = 100, default: float = 0.5) -> float:
        """Return the average of a game stat over the last `num_matches`."""
        recent_stats = getattr(self, game_stat)[-min(num_matches, len(getattr(self, game_stat))) :]
        return sum(recent_stats) / len(recent_stats) if recent_stats else default


class EloRatingEngine:
    def calculate_k_factor(self, matches_played: int, tourney_level: str) -> float:
        """Calculate K-factor based on player experience and tournament level."""
        MIN_K_FACTOR = 18.0
        MAX_K_FACTOR = 40.0

        k_factor = max(MIN_K_FACTOR, min(MAX_K_FACTOR, 400.0 / (matches_played + 1)))

        tier_multipliers = {"G": 1.1, "M": 1.0, "F": 1.0, "A": 0.9, "C": 0.6}

        return k_factor * tier_multipliers[tourney_level]

    def calculate_expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def update_player_elos(
        self,
        player_a_profile: PlayerProfile,
        player_b_profile: PlayerProfile,
        surface: str,
        tourney_level: str,
        player_a_win: float,
    ) -> None:
        """Update player elos according to this formula: https://martiningram.github.io/elo-dynamic"""
        # Update global elos
        expected_score_a = self.calculate_expected_score(
            player_a_profile.elos["global"], player_b_profile.elos["global"]
        )
        expected_score_b = 1 - expected_score_a
        k_factor_a = self.calculate_k_factor(player_a_profile.get_matches_played("global"), tourney_level)
        k_factor_b = self.calculate_k_factor(player_b_profile.get_matches_played("global"), tourney_level)
        player_a_profile.elos["global"] = player_a_profile.elos["global"] + k_factor_a * (
            player_a_win - expected_score_a
        )
        player_b_profile.elos["global"] = player_b_profile.elos["global"] + k_factor_b * (
            (1.0 - player_a_win) - expected_score_b
        )

        # Update surface-specific elos
        expected_score_a_surface = self.calculate_expected_score(
            player_a_profile.elos[surface], player_b_profile.elos[surface]
        )
        expected_score_b_surface = 1 - expected_score_a_surface
        surface_k_factor_a = self.calculate_k_factor(player_a_profile.get_matches_played(surface), tourney_level)
        surface_k_factor_b = self.calculate_k_factor(player_b_profile.get_matches_played(surface), tourney_level)
        player_a_profile.elos[surface] = player_a_profile.elos[surface] + surface_k_factor_a * (
            player_a_win - expected_score_a_surface
        )
        player_b_profile.elos[surface] = player_b_profile.elos[surface] + surface_k_factor_b * (
            (1.0 - player_a_win) - expected_score_b_surface
        )


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, PlayerProfile]]:
    """Chronological pass that produces pre-match features using only past information.

    Additional features derived:
    - rank_A, rank_B, rank_diff
    - global_elo_A, global_elo_B, global_elo_diff
    - surface_elo_A, surface_elo_B, surface_elo_diff
    - h2h_wins_A, h2h_wins_B, h2h_total_matches, h2h_diff
    """
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

        surface = str(row["surface"])

        tourney_date = pd.to_datetime(row["tourney_date"], format="%Y%m%d")
        if tourney_date != player_a_profile.last_tournament_played_date:
            player_a_profile.current_tournament_minutes_played = 0
        if tourney_date != player_b_profile.last_tournament_played_date:
            player_b_profile.current_tournament_minutes_played = 0

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

        features["player_A_surface_matches_played"].append(player_a_profile.get_matches_played(surface))
        features["player_B_surface_matches_played"].append(player_b_profile.get_matches_played(surface))
        features["surface_matches_played_diff"].append(
            player_a_profile.get_matches_played(surface) - player_b_profile.get_matches_played(surface)
        )

        one_year_ago = tourney_date - pd.DateOffset(years=1)
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

        player_a_surface_win_pct_last_100 = player_a_profile.get_recent_win_percentage(100, surface)
        player_b_surface_win_pct_last_100 = player_b_profile.get_recent_win_percentage(100, surface)
        features["player_A_surface_win_pct_last_100"].append(player_a_surface_win_pct_last_100)
        features["player_B_surface_win_pct_last_100"].append(player_b_surface_win_pct_last_100)
        features["surface_win_pct_last_100_diff"].append(
            player_a_surface_win_pct_last_100 - player_b_surface_win_pct_last_100
        )

        # Compute Elo features
        features["player_A_global_elo"].append(player_a_profile.elos["global"])
        features["player_B_global_elo"].append(player_b_profile.elos["global"])
        features["global_elo_diff"].append(player_a_profile.elos["global"] - player_b_profile.elos["global"])

        features["player_A_surface_elo"].append(player_a_profile.elos[surface])
        features["player_B_surface_elo"].append(player_b_profile.elos[surface])
        features["surface_elo_diff"].append(player_a_profile.elos[surface] - player_b_profile.elos[surface])

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
        features["hard_surface"].append(1 if surface == "Hard" else 0)
        features["clay_surface"].append(1 if surface == "Clay" else 0)
        features["grass_surface"].append(1 if surface == "Grass" else 0)

        features["best_of_5"].append(1 if row["best_of"] == 5 else 0)

        # Post-match updates to player profiles (Elo updates happen after feature capture to prevent leakage)
        elo_engine.update_player_elos(
            player_a_profile=player_a_profile,
            player_b_profile=player_b_profile,
            surface=surface,
            tourney_level=row["tourney_level"],
            player_a_win=row["player_A_win"],
        )

        player_a_profile.rank = row["player_A_rank"]
        player_b_profile.rank = row["player_B_rank"]
        player_a_profile.rank_points = row["player_A_rank_points"]
        player_b_profile.rank_points = row["player_B_rank_points"]

        player_a_profile.age = row["player_A_age"]
        player_b_profile.age = row["player_B_age"]
        player_a_profile.ht = row["player_A_ht"]
        player_b_profile.ht = row["player_B_ht"]

        player_a_profile.h2h_records[row["player_B_id"]].append(row["player_A_win"] == 1)
        player_b_profile.h2h_records[row["player_A_id"]].append(row["player_A_win"] == 0)

        player_a_profile.last_tournament_played_date = tourney_date
        player_b_profile.last_tournament_played_date = tourney_date

        player_a_profile.current_tournament_minutes_played += row["minutes"]
        player_b_profile.current_tournament_minutes_played += row["minutes"]

        player_a_profile.match_history["global"].append(_Match(date=tourney_date, won=row["player_A_win"] == 1))
        player_b_profile.match_history["global"].append(_Match(date=tourney_date, won=row["player_A_win"] == 0))
        player_a_profile.match_history[surface].append(_Match(date=tourney_date, won=row["player_A_win"] == 1))
        player_b_profile.match_history[surface].append(_Match(date=tourney_date, won=row["player_A_win"] == 0))

        player_a_profile.p_ace.append(row["player_A_ace"] / row["player_A_svpt"])
        player_b_profile.p_ace.append(row["player_B_ace"] / row["player_B_svpt"])
        player_a_profile.p_df.append(row["player_A_df"] / row["player_A_svpt"])
        player_b_profile.p_df.append(row["player_B_df"] / row["player_B_svpt"])
        player_a_profile.p_1st_in.append(row["player_A_1stIn"] / row["player_A_svpt"])
        player_b_profile.p_1st_in.append(row["player_B_1stIn"] / row["player_B_svpt"])
        player_a_profile.p_1st_won.append(row["player_A_1stWon"] / row["player_A_1stIn"])
        player_b_profile.p_1st_won.append(row["player_B_1stWon"] / row["player_B_1stIn"])
        if (row["player_A_svpt"] - row["player_A_1stIn"]) > 0:
            player_a_profile.p_2nd_won.append(row["player_A_2ndWon"] / (row["player_A_svpt"] - row["player_A_1stIn"]))
        if (row["player_B_svpt"] - row["player_B_1stIn"]) > 0:
            player_b_profile.p_2nd_won.append(row["player_B_2ndWon"] / (row["player_B_svpt"] - row["player_B_1stIn"]))
        if row["player_A_bpFaced"] > 0:
            player_a_profile.p_bp_saved.append(row["player_A_bpSaved"] / row["player_A_bpFaced"])
        if row["player_B_bpFaced"] > 0:
            player_b_profile.p_bp_saved.append(row["player_B_bpSaved"] / row["player_B_bpFaced"])
        player_a_profile.p_rp_won.append(
            (row["player_B_svpt"] - row["player_B_1stWon"] - row["player_B_2ndWon"]) / row["player_B_svpt"]
        )
        player_b_profile.p_rp_won.append(
            (row["player_A_svpt"] - row["player_A_1stWon"] - row["player_A_2ndWon"]) / row["player_A_svpt"]
        )
        if row["player_B_bpFaced"] > 0:
            player_a_profile.p_bp_won.append(
                (row["player_B_bpFaced"] - row["player_B_bpSaved"]) / row["player_B_bpFaced"]
            )
        if row["player_A_bpFaced"] > 0:
            player_b_profile.p_bp_won.append(
                (row["player_A_bpFaced"] - row["player_A_bpSaved"]) / row["player_A_bpFaced"]
            )

    return (dfc.assign(**features), player_profiles)


def get_player_profile_by_name(player_profiles: Dict[int, PlayerProfile], player_name: str) -> PlayerProfile:
    """Retrieve a PlayerProfile object by player name."""
    for profile in player_profiles.values():
        if profile.name == player_name:
            return profile

    raise ValueError(f"Player '{player_name}' not found in player profiles.")
