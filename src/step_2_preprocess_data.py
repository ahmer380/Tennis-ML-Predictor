import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from tabulate import tabulate

# Columns that are noisy and not worth keeping (many missing values, or not useful for prediction)
NOISY_COLUMNS = [
    "winner_entry",
    "winner_seed",
    "loser_entry",
    "loser_seed",
]


def preprocess_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Copy and clean the matches DataFrame into a symmetric player_A/player_B format.

    - Drops rows missing any values
    - Converts winner/loser into player_A/player_B with a random 50/50 swap
    - Returns cleaned DataFrame ready for ML (no feature engineering performed)
    """
    dfc = df.copy()
    dfc = dfc.drop(columns=NOISY_COLUMNS)
    dfc = dfc.dropna()

    dfc = dfc[(dfc["winner_ht"] >= 100) & (dfc["loser_ht"] >= 100)]
    dfc = dfc[(dfc["tourney_level"].isin(["G", "M", "F", "A", "C"]))]
    dfc = dfc[(dfc["surface"].isin(["Hard", "Clay", "Grass"]))]
    dfc = dfc[(dfc["score"] != "W/O")]

    dfc["tourney_date"] = pd.to_datetime(dfc["tourney_date"], format="%Y%m%d")

    # Convert winner/loser format into player_A/player_B with random swap to avoid bias
    dfc = dfc.rename(
        columns=lambda x: (
            x.replace("winner_", "player_A_")
            if x.startswith("winner_")
            else (
                x.replace("w_", "player_A_")
                if x.startswith("w_")
                else (
                    x.replace("loser_", "player_B_")
                    if x.startswith("loser_")
                    else x.replace("l_", "player_B_") if x.startswith("l_") else x
                )
            )
        )
    )

    swap_mask = np.random.rand(len(dfc)) < 0.5
    p1_cols = [col for col in dfc.columns if col.startswith("player_A_")]
    p2_cols = [col for col in dfc.columns if col.startswith("player_B_")]
    tmp = dfc.loc[swap_mask, p1_cols].copy()
    dfc.loc[swap_mask, p1_cols] = dfc.loc[swap_mask, p2_cols].values
    dfc.loc[swap_mask, p2_cols] = tmp.values
    dfc["player_A_win"] = (~swap_mask).astype(int)

    dfc = dfc.reset_index(drop=True)

    return dfc


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
        missing_count = int(series.isna().sum())

        if is_numeric_dtype(series) or is_datetime64_any_dtype(series):
            minimum = series.min()
            maximum = series.max()
        else:
            minimum = "N/A"
            maximum = "N/A"

        summary_rows.append(
            {
                "name": column_name,
                "data_type": str(series.dtype),
                "minimum": minimum,
                "maximum": maximum,
                "missing_rows": missing_count,
            }
        )

    summary_rows.sort(key=lambda row: row["name"])

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
