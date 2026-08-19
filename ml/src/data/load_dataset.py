import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_dataset(year_count: int = 20) -> pd.DataFrame:
    """Loads the Jeff Sackmann ATP matches dataset for the specified number of years."""
    print("\nLoading dataset...")

    atp_matches_files = sorted(DATA_DIR.glob("atp_matches_*.csv"), reverse=True)
    if len(atp_matches_files) < year_count:
        raise ValueError(
            f"Not enough data files found. Expected at least {year_count}, but found {len(atp_matches_files)}."
        )

    most_recent_years = sorted(atp_matches_files[:year_count])
    dfs = [pd.read_csv(filepath) for filepath in most_recent_years]

    return pd.concat(dfs, ignore_index=True)
