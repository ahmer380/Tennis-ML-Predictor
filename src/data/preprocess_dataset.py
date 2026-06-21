import numpy as np
import pandas as pd

# Columns that are noisy and not worth keeping (many missing values, or not useful for prediction)
NOISY_COLUMNS = [
    "winner_entry",
    "winner_seed",
    "loser_entry",
    "loser_seed",
]


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Copy and clean the matches DataFrame into a symmetric player_A/player_B format."""
    print("\nPreprocessing dataset...")

    dfc = df.copy()
    dfc = dfc.drop(columns=NOISY_COLUMNS)
    dfc = dfc.dropna()

    # Miscellaneous filters to remove outlier matches
    dfc = dfc[(dfc["winner_ht"] >= 100) & (dfc["loser_ht"] >= 100)]
    dfc = dfc[(dfc["w_svpt"] > 0) & (dfc["l_svpt"] > 0)]
    dfc = dfc[dfc["tourney_level"].isin(["G", "M", "F", "A", "C"])]
    dfc = dfc[dfc["surface"].isin(["Hard", "Clay", "Grass"])]
    dfc = dfc[~dfc["score"].str.contains("W/O|RET|DEF", case=False, na=False)]

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
