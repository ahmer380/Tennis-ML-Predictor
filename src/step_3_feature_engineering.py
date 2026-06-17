from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

BASE_ELO = 1500.0


@dataclass
class PlayerState:
    name: str
    elos: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: BASE_ELO))
    match_history: Dict[str, List[bool]] = field(default_factory=lambda: defaultdict(list))
    last_tournament_played_date: pd.Timestamp = None
    current_tournament_minutes_played: int = 0
    h2h_records: Dict[int, List[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )  # key=opponent_id, value=win/loss record

    def get_matches_played(self, surface: str = "global") -> int:
        """Return the number of matches played on a given surface."""
        return len(self.match_history[surface])

    def get_recent_win_percentage(self, num_matches: int, surface: str = "global") -> float:
        """Return the win percentage over the last `num_matches` on a given surface."""
        recent_matches = self.match_history[surface][-min(num_matches, len(self.match_history[surface])) :]
        return sum(recent_matches) / len(recent_matches) if recent_matches else 0.0


class EloRatingEngine:
    def calculate_k_factor(self, matches_played: int, tourney_level: str) -> float:
        """Calculate K-factor based on player experience and tournament level."""
        MIN_K_FACTOR = 18.0
        MAX_K_FACTOR = 40.0

        k_factor = max(MIN_K_FACTOR, min(MAX_K_FACTOR, 400.0 / (matches_played + 1)))

        tier_multipliers = {"G": 1.15, "M": 1.0, "F": 1.0, "A": 0.9, "C": 0.6}

        return k_factor * tier_multipliers[tourney_level]

    def calculate_expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def update_player_elos(
        self,
        player_a_state: PlayerState,
        player_b_state: PlayerState,
        match_date: pd.Timestamp,
        surface: str,
        tourney_level: str,
        player_a_win: float,
    ) -> None:
        """Update player elos according to this formula: https://martiningram.github.io/elo-dynamic"""
        # Update global elos
        expected_score_a = self.calculate_expected_score(player_a_state.elos["global"], player_b_state.elos["global"])
        expected_score_b = 1 - expected_score_a
        k_factor_a = self.calculate_k_factor(player_a_state.get_matches_played("global"), tourney_level)
        k_factor_b = self.calculate_k_factor(player_b_state.get_matches_played("global"), tourney_level)
        player_a_state.elos["global"] = player_a_state.elos["global"] + k_factor_a * (player_a_win - expected_score_a)
        player_b_state.elos["global"] = player_b_state.elos["global"] + k_factor_b * (
            (1.0 - player_a_win) - expected_score_b
        )

        # Update surface-specific elos
        expected_score_a_surface = self.calculate_expected_score(
            player_a_state.elos[surface], player_b_state.elos[surface]
        )
        expected_score_b_surface = 1 - expected_score_a_surface
        surface_k_factor_a = self.calculate_k_factor(player_a_state.get_matches_played(surface), tourney_level)
        surface_k_factor_b = self.calculate_k_factor(player_b_state.get_matches_played(surface), tourney_level)
        player_a_state.elos[surface] = player_a_state.elos[surface] + surface_k_factor_a * (
            player_a_win - expected_score_a_surface
        )
        player_b_state.elos[surface] = player_b_state.elos[surface] + surface_k_factor_b * (
            (1.0 - player_a_win) - expected_score_b_surface
        )


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, PlayerState]]:
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

    # key=player_id, value=PlayerState object
    player_states: Dict[int, PlayerState] = {}
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
        "player_A_surface_win_pct_last_10": [],
        "player_B_surface_win_pct_last_10": [],
        "surface_win_pct_last_10_diff": [],
        "player_A_global_elo": [],
        "player_B_global_elo": [],
        "global_elo_diff": [],
        "player_A_surface_elo": [],
        "player_B_surface_elo": [],
        "surface_elo_diff": [],
        "player_A_h2h_wins": [],
        "player_B_h2h_wins": [],
        "h2h_diff": [],
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
        if row["player_A_id"] not in player_states:
            player_states[row["player_A_id"]] = PlayerState(name=row["player_A_name"])
        if row["player_B_id"] not in player_states:
            player_states[row["player_B_id"]] = PlayerState(name=row["player_B_name"])

        player_a_state = player_states[row["player_A_id"]]
        player_b_state = player_states[row["player_B_id"]]

        surface = str(row["surface"])

        tourney_date = pd.to_datetime(row["tourney_date"], format="%Y%m%d")
        if tourney_date != player_a_state.last_tournament_played_date:
            player_a_state.current_tournament_minutes_played = 0
        if tourney_date != player_b_state.last_tournament_played_date:
            player_b_state.current_tournament_minutes_played = 0

        # Compute feature differences
        features["rank_diff"].append(row["player_A_rank"] - row["player_B_rank"])
        features["rank_points_diff"].append(row["player_A_rank_points"] - row["player_B_rank_points"])
        features["age_diff"].append(row["player_A_age"] - row["player_B_age"])
        features["ht_diff"].append(row["player_A_ht"] - row["player_B_ht"])

        # Compute experience features
        features["player_A_global_matches_played"].append(player_a_state.get_matches_played("global"))
        features["player_B_global_matches_played"].append(player_b_state.get_matches_played("global"))
        features["global_matches_played_diff"].append(
            player_a_state.get_matches_played("global") - player_b_state.get_matches_played("global")
        )

        features["player_A_surface_matches_played"].append(player_a_state.get_matches_played(surface))
        features["player_B_surface_matches_played"].append(player_b_state.get_matches_played(surface))
        features["surface_matches_played_diff"].append(
            player_a_state.get_matches_played(surface) - player_b_state.get_matches_played(surface)
        )

        # Compute form features
        player_a_global_win_pct_last_10 = player_a_state.get_recent_win_percentage(10, "global")
        player_b_global_win_pct_last_10 = player_b_state.get_recent_win_percentage(10, "global")
        features["player_A_global_win_pct_last_10"].append(player_a_global_win_pct_last_10)
        features["player_B_global_win_pct_last_10"].append(player_b_global_win_pct_last_10)
        features["global_win_pct_last_10_diff"].append(
            player_a_global_win_pct_last_10 - player_b_global_win_pct_last_10
        )

        player_a_global_win_pct_last_25 = player_a_state.get_recent_win_percentage(25, "global")
        player_b_global_win_pct_last_25 = player_b_state.get_recent_win_percentage(25, "global")
        features["player_A_global_win_pct_last_25"].append(player_a_global_win_pct_last_25)
        features["player_B_global_win_pct_last_25"].append(player_b_global_win_pct_last_25)
        features["global_win_pct_last_25_diff"].append(
            player_a_global_win_pct_last_25 - player_b_global_win_pct_last_25
        )

        player_a_global_win_pct_last_50 = player_a_state.get_recent_win_percentage(50, "global")
        player_b_global_win_pct_last_50 = player_b_state.get_recent_win_percentage(50, "global")
        features["player_A_global_win_pct_last_50"].append(player_a_global_win_pct_last_50)
        features["player_B_global_win_pct_last_50"].append(player_b_global_win_pct_last_50)
        features["global_win_pct_last_50_diff"].append(
            player_a_global_win_pct_last_50 - player_b_global_win_pct_last_50
        )

        player_a_global_win_pct_last_100 = player_a_state.get_recent_win_percentage(100, "global")
        player_b_global_win_pct_last_100 = player_b_state.get_recent_win_percentage(100, "global")
        features["player_A_global_win_pct_last_100"].append(player_a_global_win_pct_last_100)
        features["player_B_global_win_pct_last_100"].append(player_b_global_win_pct_last_100)
        features["global_win_pct_last_100_diff"].append(
            player_a_global_win_pct_last_100 - player_b_global_win_pct_last_100
        )

        player_a_surface_win_pct_last_10 = player_a_state.get_recent_win_percentage(10, surface)
        player_b_surface_win_pct_last_10 = player_b_state.get_recent_win_percentage(10, surface)
        features["player_A_surface_win_pct_last_10"].append(player_a_surface_win_pct_last_10)
        features["player_B_surface_win_pct_last_10"].append(player_b_surface_win_pct_last_10)
        features["surface_win_pct_last_10_diff"].append(
            player_a_surface_win_pct_last_10 - player_b_surface_win_pct_last_10
        )

        # Compute Elo features
        features["player_A_global_elo"].append(player_a_state.elos["global"])
        features["player_B_global_elo"].append(player_b_state.elos["global"])
        features["global_elo_diff"].append(player_a_state.elos["global"] - player_b_state.elos["global"])

        features["player_A_surface_elo"].append(player_a_state.elos[surface])
        features["player_B_surface_elo"].append(player_b_state.elos[surface])
        features["surface_elo_diff"].append(player_a_state.elos[surface] - player_b_state.elos[surface])

        # Compute head-to-head features
        h2h_wins_a = player_a_state.h2h_records[row["player_B_id"]].count(True)
        h2h_wins_b = player_b_state.h2h_records[row["player_A_id"]].count(True)
        features["player_A_h2h_wins"].append(h2h_wins_a)
        features["player_B_h2h_wins"].append(h2h_wins_b)
        features["h2h_diff"].append(h2h_wins_a - h2h_wins_b)

        # Compute tournament fatigue features
        features["player_A_tournament_minutes"].append(player_a_state.current_tournament_minutes_played)
        features["player_B_tournament_minutes"].append(player_b_state.current_tournament_minutes_played)
        features["tournament_minutes_diff"].append(
            player_a_state.current_tournament_minutes_played - player_b_state.current_tournament_minutes_played
        )

        # Compute match context features
        features["hard_surface"].append(1 if surface == "Hard" else 0)
        features["clay_surface"].append(1 if surface == "Clay" else 0)
        features["grass_surface"].append(1 if surface == "Grass" else 0)

        features["best_of_5"].append(1 if row["best_of"] == 5 else 0)

        # Post-match updates to player states (Elo updates happen after feature capture to prevent leakage)
        elo_engine.update_player_elos(
            player_a_state=player_a_state,
            player_b_state=player_b_state,
            match_date=tourney_date,
            surface=surface,
            tourney_level=row["tourney_level"],
            player_a_win=row["player_A_win"],
        )

        player_a_state.h2h_records[row["player_B_id"]].append(row["player_A_win"] == 1)
        player_b_state.h2h_records[row["player_A_id"]].append(row["player_A_win"] == 0)

        player_a_state.last_tournament_played_date = tourney_date
        player_b_state.last_tournament_played_date = tourney_date

        player_a_state.current_tournament_minutes_played += row["minutes"]
        player_b_state.current_tournament_minutes_played += row["minutes"]

        player_a_state.match_history["global"].append(row["player_A_win"] == 1)
        player_b_state.match_history["global"].append(row["player_A_win"] == 0)
        player_a_state.match_history[surface].append(row["player_A_win"] == 1)
        player_b_state.match_history[surface].append(row["player_A_win"] == 0)

    return (dfc.assign(**features), player_states)


def audit_player_states(player_states: Dict[int, PlayerState]) -> None:
    """Print the top 50 players by global Elo, including ranks for all Elo types.
    Use website as point of comparison: https://www.tennisabstract.com/reports/atp_elo_ratings.html
    """
    print(f"There are {len(player_states)} unique players in the dataset.\n")
    player_rows = []
    for player_state in player_states.values():
        player_rows.append(
            {
                "player_name": player_state.name,
                "global": float(player_state.elos["global"]),
                "hard": float(player_state.elos["Hard"]),
                "clay": float(player_state.elos["Clay"]),
                "grass": float(player_state.elos["Grass"]),
            }
        )

    # Compute 1-based ranks for each Elo type.
    elo_types = ["global", "hard", "clay", "grass"]
    ranks = {elo_type: {} for elo_type in elo_types}
    for elo_type in elo_types:
        sorted_rows = sorted(player_rows, key=lambda row: row[elo_type], reverse=True)
        for rank, row in enumerate(sorted_rows, start=1):
            ranks[elo_type][row["player_name"]] = rank

    top_players = sorted(player_rows, key=lambda row: row["global"], reverse=True)[:50]

    table_rows = []
    for row in top_players:
        player_name = row["player_name"]
        table_rows.append(
            [
                ranks["global"][player_name],
                player_name,
                row["global"],
                row["hard"],
                ranks["hard"][player_name],
                row["clay"],
                ranks["clay"][player_name],
                row["grass"],
                ranks["grass"][player_name],
            ]
        )

    headers = [
        "rank",
        "player_name",
        "global_elo",
        "hard_elo",
        "hard_elo_rank",
        "clay_elo",
        "clay_elo_rank",
        "grass_elo",
        "grass_elo_rank",
    ]

    print(tabulate(table_rows, headers=headers, tablefmt="github", floatfmt=".2f"))


def audit_player_h2h(df_features: pd.DataFrame, player_a_name: str, player_b_name: str) -> None:
    h2h_matches = df_features[
        ((df_features["player_A_name"] == player_a_name) & (df_features["player_B_name"] == player_b_name))
        | ((df_features["player_A_name"] == player_b_name) & (df_features["player_B_name"] == player_a_name))
    ]

    print(f"\n {player_a_name} vs {player_b_name} matches with engineered features:\n")
    print(
        h2h_matches[
            [
                "tourney_name",
                "player_A_name",
                "player_B_name",
                "player_A_global_elo",
                "player_B_global_elo",
                "player_A_surface_elo",
                "player_B_surface_elo",
                "player_A_h2h_wins",
                "player_B_h2h_wins",
                "player_A_win",
            ]
        ]
    )


def audit_match(
    df_features: pd.DataFrame, player_a_name: str, player_b_name: str, tournament_name: str, year: int
) -> None:
    match = df_features[
        (
            ((df_features["player_A_name"] == player_a_name) & (df_features["player_B_name"] == player_b_name))
            | ((df_features["player_A_name"] == player_b_name) & (df_features["player_B_name"] == player_a_name))
        )
        & (df_features["tourney_name"] == tournament_name)
        & (pd.to_datetime(df_features["tourney_date"], format="%Y%m%d").dt.year == year)
    ]

    if match.empty:
        print(f"No match found between {player_a_name} and {player_b_name} in {tournament_name} {year}.")

    for col, value in match.iloc[0].items():
        print(f"{col}: {value}")


def audit_player_tournament_run(df_features: pd.DataFrame, player_name: str, tournament_name: str, year: int) -> None:
    tournament_matches = df_features[
        ((df_features["player_A_name"] == player_name) | (df_features["player_B_name"] == player_name))
        & (df_features["tourney_name"] == tournament_name)
        & (pd.to_datetime(df_features["tourney_date"], format="%Y%m%d").dt.year == year)
    ]

    print(f"\n {player_name} matches in {tournament_name} {year} with engineered features:\n")
    print(
        tournament_matches[
            [
                "tourney_name",
                "player_A_name",
                "player_B_name",
                "player_A_global_elo",
                "player_B_global_elo",
                "player_A_surface_elo",
                "player_B_surface_elo",
                "player_A_h2h_wins",
                "player_B_h2h_wins",
                "player_A_tournament_minutes",
                "player_B_tournament_minutes",
                "player_A_win",
            ]
        ]
    )


def plot_player_career_elo_trajectory(df_features: pd.DataFrame, player_name: str) -> None:
    """Plot the career Elo trajectory of a specific player."""
    player_matches = df_features[
        (df_features["player_A_name"] == player_name) | (df_features["player_B_name"] == player_name)
    ]

    global_elo_values = player_matches.apply(
        lambda row: row["player_A_global_elo"] if row["player_A_name"] == player_name else row["player_B_global_elo"],
        axis=1,
    )
    global_elo_dates = pd.to_datetime(player_matches["tourney_date"], format="%Y%m%d")

    hard_surface_matches = player_matches[player_matches["surface"] == "Hard"]
    hard_elo_values = hard_surface_matches.apply(
        lambda row: row["player_A_surface_elo"] if row["player_A_name"] == player_name else row["player_B_surface_elo"],
        axis=1,
    )
    hard_elo_dates = pd.to_datetime(hard_surface_matches["tourney_date"], format="%Y%m%d")

    clay_surface_matches = player_matches[player_matches["surface"] == "Clay"]
    clay_elo_values = clay_surface_matches.apply(
        lambda row: row["player_A_surface_elo"] if row["player_A_name"] == player_name else row["player_B_surface_elo"],
        axis=1,
    )
    clay_elo_dates = pd.to_datetime(clay_surface_matches["tourney_date"], format="%Y%m%d")

    grass_surface_matches = player_matches[player_matches["surface"] == "Grass"]
    grass_elo_values = grass_surface_matches.apply(
        lambda row: row["player_A_surface_elo"] if row["player_A_name"] == player_name else row["player_B_surface_elo"],
        axis=1,
    )
    grass_elo_dates = pd.to_datetime(grass_surface_matches["tourney_date"], format="%Y%m%d")

    plt.figure(figsize=(12, 6))
    plt.plot(global_elo_dates, global_elo_values, label="Global Elo", color="black")
    plt.plot(hard_elo_dates, hard_elo_values, label="Hard Surface Elo", color="blue", linestyle="--")
    plt.plot(clay_elo_dates, clay_elo_values, label="Clay Surface Elo", color="orange", linestyle="--")
    plt.plot(grass_elo_dates, grass_elo_values, label="Grass Surface Elo", color="green", linestyle="--")
    plt.title(f"Career Elo Trajectory of {player_name}")
    plt.xlabel("Date")
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
