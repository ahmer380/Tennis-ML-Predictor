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
    global_elo: float = BASE_ELO
    surface_elos: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: BASE_ELO))
    last_tournament_played_date: pd.Timestamp = None
    current_tournament_minutes_played: int = 0
    h2h_records: Dict[int, List[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )  # key=opponent_id, value=win/loss record


class EloRatingEngine:
    K_FACTOR = 24.0  # standard K-factor for tennis Elo
    INACTIVITY_THRESHOLD_DAYS = 100  # days after which Elo starts decaying towards baseline
    INACTIVITY_DECAY_RATE = 0.002  # daily decay rate for inactivity beyond threshold

    # TODO:Increase K-factor based on player inactivity, number of matches played, or tournament importance (e.g., Grand Slams)

    def calculate_expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def apply_inactivity_decay(self, player_state: PlayerState, match_date: pd.Timestamp) -> None:
        """Update Elo for inactivity beyond threshold days since last tournament towards the baseline Elo."""
        days_since_last_tournament = (
            0
            if player_state.last_tournament_played_date is None
            else (match_date - player_state.last_tournament_played_date).days
        )
        if days_since_last_tournament > self.INACTIVITY_THRESHOLD_DAYS:
            days_inactive = days_since_last_tournament - self.INACTIVITY_THRESHOLD_DAYS
            decay = (1.0 - self.INACTIVITY_DECAY_RATE) ** days_inactive

            player_state.global_elo = BASE_ELO + ((player_state.global_elo - BASE_ELO) * decay)
            for surface in player_state.surface_elos:
                player_state.surface_elos[surface] = BASE_ELO + (
                    (player_state.surface_elos[surface] - BASE_ELO) * decay
                )

    def update_player_elos(
        self,
        player_a_state: PlayerState,
        player_b_state: PlayerState,
        match_date: pd.Timestamp,
        surface: str,
        player_a_win: float,
    ) -> None:
        """Update player elos according to this formula: https://martiningram.github.io/elo-dynamic"""

        # Apply inactivity decay before calculating expected scores
        self.apply_inactivity_decay(player_a_state, match_date)
        self.apply_inactivity_decay(player_b_state, match_date)

        # Update global elos
        expected_score_a = self.calculate_expected_score(player_a_state.global_elo, player_b_state.global_elo)
        expected_score_b = 1 - expected_score_a
        player_a_state.global_elo = player_a_state.global_elo + self.K_FACTOR * (player_a_win - expected_score_a)
        player_b_state.global_elo = player_b_state.global_elo + self.K_FACTOR * (
            (1.0 - player_a_win) - expected_score_b
        )

        # Update surface-specific elos
        expected_score_a_surface = self.calculate_expected_score(
            player_a_state.surface_elos[surface], player_b_state.surface_elos[surface]
        )
        expected_score_b_surface = 1 - expected_score_a_surface
        player_a_state.surface_elos[surface] = player_a_state.surface_elos[surface] + self.K_FACTOR * (
            player_a_win - expected_score_a_surface
        )
        player_b_state.surface_elos[surface] = player_b_state.surface_elos[surface] + self.K_FACTOR * (
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

    # Lists to collect features in order
    features = {
        "rank_A": [],
        "rank_B": [],
        "rank_diff": [],
        "global_elo_A": [],
        "global_elo_B": [],
        "global_elo_diff": [],
        "surface_elo_A": [],
        "surface_elo_B": [],
        "surface_elo_diff": [],
        "h2h_wins_A": [],
        "h2h_wins_B": [],
        "h2h_diff": [],
        "age_A": [],
        "age_B": [],
        "age_diff": [],
        "tournament_minutes_A": [],
        "tournament_minutes_B": [],
        "tournament_minutes_diff": [],
        "hard_surface": [],
        "clay_surface": [],
        "grass_surface": [],
        "best_of_5": [],
        "player_A_win": [],
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

        # Populate feature lists (Elo values are pre-match)
        features["rank_A"].append(row["player_A_rank"])
        features["rank_B"].append(row["player_B_rank"])
        features["rank_diff"].append(row["player_A_rank"] - row["player_B_rank"])

        features["global_elo_A"].append(player_a_state.global_elo)
        features["global_elo_B"].append(player_b_state.global_elo)
        features["global_elo_diff"].append(player_a_state.global_elo - player_b_state.global_elo)

        features["surface_elo_A"].append(player_a_state.surface_elos[surface])
        features["surface_elo_B"].append(player_b_state.surface_elos[surface])
        features["surface_elo_diff"].append(player_a_state.surface_elos[surface] - player_b_state.surface_elos[surface])

        h2h_wins_a = player_a_state.h2h_records[row["player_B_id"]].count(True)
        h2h_wins_b = player_b_state.h2h_records[row["player_A_id"]].count(True)
        features["h2h_wins_A"].append(h2h_wins_a)
        features["h2h_wins_B"].append(h2h_wins_b)
        features["h2h_diff"].append(h2h_wins_a - h2h_wins_b)

        features["age_A"].append(row["player_A_age"])
        features["age_B"].append(row["player_B_age"])
        features["age_diff"].append(row["player_A_age"] - row["player_B_age"])

        features["tournament_minutes_A"].append(player_a_state.current_tournament_minutes_played)
        features["tournament_minutes_B"].append(player_b_state.current_tournament_minutes_played)
        features["tournament_minutes_diff"].append(
            player_a_state.current_tournament_minutes_played - player_b_state.current_tournament_minutes_played
        )

        features["hard_surface"].append(1 if surface == "Hard" else 0)
        features["clay_surface"].append(1 if surface == "Clay" else 0)
        features["grass_surface"].append(1 if surface == "Grass" else 0)

        features["best_of_5"].append(1 if row["best_of"] == 5 else 0)

        features["player_A_win"].append(1 if row["player_A_win"] == 1 else 0)

        # Post-match updates to player states (Elo updates happen after feature capture to prevent leakage)
        elo_engine.update_player_elos(player_a_state, player_b_state, tourney_date, surface, row["player_A_win"])

        player_a_state.h2h_records[row["player_B_id"]].append(row["player_A_win"] == 1)
        player_b_state.h2h_records[row["player_A_id"]].append(row["player_A_win"] == 0)

        player_a_state.last_tournament_played_date = tourney_date
        player_b_state.last_tournament_played_date = tourney_date

        player_a_state.current_tournament_minutes_played += row["minutes"]
        player_b_state.current_tournament_minutes_played += row["minutes"]

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
                "global": float(player_state.global_elo),
                "hard": float(player_state.surface_elos["Hard"]),
                "clay": float(player_state.surface_elos["Clay"]),
                "grass": float(player_state.surface_elos["Grass"]),
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
                "player_A_name",
                "player_B_name",
                "tourney_name",
                "global_elo_A",
                "global_elo_B",
                "surface_elo_A",
                "surface_elo_B",
                "h2h_wins_A",
                "h2h_wins_B",
                "player_A_win",
            ]
        ]
    )


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
                "player_A_name",
                "player_B_name",
                "tourney_name",
                "global_elo_A",
                "global_elo_B",
                "h2h_wins_A",
                "h2h_wins_B",
                "tournament_minutes_A",
                "tournament_minutes_B",
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
        lambda row: row["global_elo_A"] if row["player_A_name"] == player_name else row["global_elo_B"], axis=1
    )
    global_elo_dates = pd.to_datetime(player_matches["tourney_date"], format="%Y%m%d")

    hard_surface_matches = player_matches[player_matches["surface"] == "Hard"]
    hard_elo_values = hard_surface_matches.apply(
        lambda row: row["surface_elo_A"] if row["player_A_name"] == player_name else row["surface_elo_B"], axis=1
    )
    hard_elo_dates = pd.to_datetime(hard_surface_matches["tourney_date"], format="%Y%m%d")

    clay_surface_matches = player_matches[player_matches["surface"] == "Clay"]
    clay_elo_values = clay_surface_matches.apply(
        lambda row: row["surface_elo_A"] if row["player_A_name"] == player_name else row["surface_elo_B"], axis=1
    )
    clay_elo_dates = pd.to_datetime(clay_surface_matches["tourney_date"], format="%Y%m%d")

    grass_surface_matches = player_matches[player_matches["surface"] == "Grass"]
    grass_elo_values = grass_surface_matches.apply(
        lambda row: row["surface_elo_A"] if row["player_A_name"] == player_name else row["surface_elo_B"], axis=1
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
