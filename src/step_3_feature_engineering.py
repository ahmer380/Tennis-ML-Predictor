from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

BASE_ELO = 1500.0


@dataclass
class PlayerState:
    name: str
    global_elo: float = BASE_ELO
    surface_elos: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: BASE_ELO))
    last_tournament_played_date: pd.Timestamp = None
    last_10_win_record: Dict[int, List[bool]] = field(
        default_factory=lambda: defaultdict(lambda: [False] * 10)
    )  # key=opponent_id, value=win/loss record in last 10 matches


class EloRatingEngine:
    K_FACTOR = 24.0  # standard K-factor for tennis Elo
    INACTIVITY_THRESHOLD_DAYS = 100  # days after which Elo starts decaying towards baseline
    INACTIVITY_DECAY_RATE = 0.002  # daily decay rate for inactivity beyond threshold

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

        h2h_wins_a = player_a_state.last_10_win_record[row["player_B_id"]].count(True)
        h2h_wins_b = player_b_state.last_10_win_record[row["player_A_id"]].count(True)
        features["h2h_wins_A"].append(h2h_wins_a)
        features["h2h_wins_B"].append(h2h_wins_b)
        features["h2h_diff"].append(h2h_wins_a - h2h_wins_b)

        features["age_A"].append(row["player_A_age"])
        features["age_B"].append(row["player_B_age"])
        features["age_diff"].append(row["player_A_age"] - row["player_B_age"])

        features["hard_surface"].append(1 if surface == "Hard" else 0)
        features["clay_surface"].append(1 if surface == "Clay" else 0)
        features["grass_surface"].append(1 if surface == "Grass" else 0)

        features["best_of_5"].append(1 if row["best_of"] == 5 else 0)

        features["player_A_win"].append(1 if row["player_A_win"] == 1 else 0)

        # Post-match updates to player states (Elo updates happen after feature capture to prevent leakage)
        elo_engine.update_player_elos(player_a_state, player_b_state, tourney_date, surface, row["player_A_win"])

        player_a_state.last_10_win_record[row["player_B_id"]].append(row["player_A_win"] == 1)
        player_b_state.last_10_win_record[row["player_A_id"]].append(row["player_A_win"] == 0)
        player_a_state.last_10_win_record[row["player_B_id"]].pop(0)
        player_b_state.last_10_win_record[row["player_A_id"]].pop(0)

        player_a_state.last_tournament_played_date = tourney_date
        player_b_state.last_tournament_played_date = tourney_date

    return (dfc.assign(**features), player_states)


def audit_player_state(player_states: Dict[int, PlayerState]) -> None:
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
