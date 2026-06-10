import pandas as pd

def predict(feature: pd.Series) -> int:
    """Predict the winner of a match based on global ELO ratings."""
    if feature["global_elo_A"] > feature["global_elo_B"]:
        return 1
    else:
        return 0
