import matplotlib.pyplot as plt
import pandas as pd


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

    # Plotting the Elo trajectory by date
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

    # Plotting the Elo trajectory by match number
    plt.figure(figsize=(12, 6))
    plt.plot([i for i in range(len(global_elo_dates))], global_elo_values, label="Global Elo", color="black")
    plt.axhline(y=global_elo_values.iloc[-1], color="black")
    plt.plot(
        [i for i in range(len(hard_elo_dates))], hard_elo_values, label="Hard Surface Elo", color="blue", linestyle="--"
    )
    plt.axhline(y=hard_elo_values.iloc[-1], color="blue", linestyle="--")
    plt.plot(
        [i for i in range(len(clay_elo_dates))],
        clay_elo_values,
        label="Clay Surface Elo",
        color="orange",
        linestyle="--",
    )
    plt.axhline(y=clay_elo_values.iloc[-1], color="orange", linestyle="--")
    plt.plot(
        [i for i in range(len(grass_elo_dates))],
        grass_elo_values,
        label="Grass Surface Elo",
        color="green",
        linestyle="--",
    )
    plt.axhline(y=grass_elo_values.iloc[-1], color="green", linestyle="--")
    plt.title(f"Career Elo Trajectory of {player_name}")
    plt.xlabel("Match Number")
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
