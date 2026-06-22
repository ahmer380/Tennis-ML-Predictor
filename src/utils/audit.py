from typing import Dict

import pandas as pd
from tabulate import tabulate

from src.feature.player_profile import PlayerProfile


def audit_dataset(df: pd.DataFrame) -> None:
    """Summarise the preprocessed dataset."""
    print(f"\nShape of dataset: {df.shape}\n")
    if "player_A_win" in df.columns:
        print(df["player_A_win"].value_counts(normalize=True))
        print()

    # Audit each column
    summary_rows = []
    for column_name in df.columns:
        series = df[column_name]
        summary_rows.append(
            {
                "name": column_name,
                "data_type": str(series.dtype),
                "minimum": series.min(),
                "maximum": series.max(),
                "missing_rows": int(series.isna().sum()),
            }
        )

    print("Dataset columns summary:")
    print(tabulate(summary_rows, headers="keys", tablefmt="github", showindex=False))

    # Audit key categorical columns by grouping
    for group_column, title in (
        ("surface", "Surface"),
        ("tourney_level", "Tourney level"),
    ):
        if group_column not in df.columns:
            continue

        grouped_rows = []

        for value, group_df in df.groupby(group_column, dropna=False):
            grouped_rows.append(
                {
                    "type": value,
                    "total_rows": len(group_df),
                    "missing_rows": group_df.isna().any(axis=1).sum(),
                }
            )

        grouped_rows.sort(key=lambda row: (-row["total_rows"], str(row["type"])))

        print(f"\n{title} summary:")
        print(tabulate(grouped_rows, headers="keys", tablefmt="github", showindex=False))


def audit_player_profiles(player_profiles: Dict[int, PlayerProfile]) -> None:
    """Print the top 50 players by global Elo, including ranks for all Elo types.
    Use website as point of comparison: https://www.tennisabstract.com/reports/atp_elo_ratings.html
    """
    print(f"There are {len(player_profiles)} unique players in the dataset.\n")
    player_rows = []
    for player_profile in player_profiles.values():
        player_rows.append(
            {
                "player_name": player_profile.name,
                "global": float(player_profile.elos["global"]),
                "hard": float(player_profile.elos["Hard"]),
                "clay": float(player_profile.elos["Clay"]),
                "grass": float(player_profile.elos["Grass"]),
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
