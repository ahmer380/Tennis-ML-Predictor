from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate


def expected_score(elo_a: float, elo_b: float) -> float:
    """Return expected score for player A vs player B."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_ratings(
    elo_a: float, elo_b: float, player_a_win: float, k_factor: float
) -> Tuple[float, float]:
    """Update and return new ratings (elo_a_new, elo_b_new), according to this formula: https://martiningram.github.io/elo-dynamic"""
    expected_score_a = expected_score(elo_a, elo_b)
    expected_score_b = 1 - expected_score_a  # equal to expected_score(elo_b, elo_a)
    elo_a_new = elo_a + k_factor * (player_a_win - expected_score_a)
    elo_b_new = elo_b + k_factor * ((1.0 - player_a_win) - expected_score_b)

    return elo_a_new, elo_b_new


def engineer_features(
    df: pd.DataFrame, k_factor: float = 32.0
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """Chronological pass that produces pre-match features using only past information.

    Additional features derived:
    - rank_A, rank_B, rank_diff
    - global_elo_A, global_elo_B, global_elo_diff
    - surface_elo_A, surface_elo_B, surface_elo_diff
    - recent_win_pct_A, recent_win_pct_B, recent_win_pct_diff (placeholder)
    """
    dfc = df.copy()

    # Sort matches chronologically (oldest to newest) to prevent future data leakage in feature engineering
    dfc = dfc.sort_values(
        by=["tourney_date", "match_num"], kind="mergesort"
    ).reset_index(drop=True)

    # key=(player_id, player_name), value=dict[key=global|surface, value=elo_rating]
    elo_ratings: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: 1500.0)
    )

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
        player_a = (int(row["player_A_id"]), str(row["player_A_name"]))
        player_b = (int(row["player_B_id"]), str(row["player_B_name"]))
        surface = str(row["surface"])

        # Pre-match Elo retrieval (unseen players default to 1500)
        global_elo_a = elo_ratings[player_a]["global"]
        global_elo_b = elo_ratings[player_b]["global"]
        surface_elo_a = elo_ratings[player_a][surface]
        surface_elo_b = elo_ratings[player_b][surface]

        # Populate feature lists (Elo values are pre-match)
        features["rank_A"].append(row["player_A_rank"])
        features["rank_B"].append(row["player_B_rank"])
        features["rank_diff"].append(row["player_A_rank"] - row["player_B_rank"])

        features["global_elo_A"].append(global_elo_a)
        features["global_elo_B"].append(global_elo_b)
        features["global_elo_diff"].append(global_elo_a - global_elo_b)

        features["surface_elo_A"].append(surface_elo_a)
        features["surface_elo_B"].append(surface_elo_b)
        features["surface_elo_diff"].append(surface_elo_a - surface_elo_b)

        # TODO: Placeholder rolling features (implement later)
        features["h2h_wins_A"].append(np.nan)
        features["h2h_wins_B"].append(np.nan)
        features["h2h_diff"].append(np.nan)

        features["age_A"].append(row["player_A_age"])
        features["age_B"].append(row["player_B_age"])
        features["age_diff"].append(row["player_A_age"] - row["player_B_age"])

        features["hard_surface"].append(1 if surface == "Hard" else 0)
        features["clay_surface"].append(1 if surface == "Clay" else 0)
        features["grass_surface"].append(1 if surface == "Grass" else 0)

        features["best_of_5"].append(1 if row["best_of"] == 5 else 0)

        features["player_A_win"].append(1 if row["player_A_win"] == 1 else 0)

        # Update Elo ratings using the match result
        global_elo_a, global_elo_b = update_ratings(
            global_elo_a, global_elo_b, row["player_A_win"], k_factor
        )
        elo_ratings[player_a]["global"] = global_elo_a
        elo_ratings[player_b]["global"] = global_elo_b

        surface_elo_a, surface_elo_b = update_ratings(
            surface_elo_a, surface_elo_b, row["player_A_win"], k_factor
        )
        elo_ratings[player_a][surface] = surface_elo_a
        elo_ratings[player_b][surface] = surface_elo_b

    return (dfc.assign(**features), elo_ratings)


def audit_elo_ratings(elo_ratings: Dict[Tuple[int, str], Dict[str, float]]):
    """Print the top 50 players by global Elo, including ranks for all Elo types.
    Use website as point of comparison: https://www.tennisabstract.com/reports/atp_elo_ratings.html
    """
    player_rows = []
    for (_, player_name), ratings in elo_ratings.items():
        player_rows.append(
            {
                "player_name": player_name,
                "global": float(ratings["global"]),
                "hard": float(ratings["Hard"]),
                "clay": float(ratings["Clay"]),
                "grass": float(ratings["Grass"]),
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
