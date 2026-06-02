from datetime import date
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"


def download_dataset(year_count: int = 10) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for year in range(date.today().year - year_count, date.today().year + 1):
            url = f"{BASE_URL}/atp_matches_{year}.csv"
            response = requests.get(url)
            filepath = OUTPUT_DIR / f"atp_matches_{year}.csv"
            filepath.write_bytes(response.content)

        # download schema file for information about the columns
        url = f"{BASE_URL}/matches_data_dictionary.txt"
        response = requests.get(url)
        filepath = OUTPUT_DIR / "matches_data_dictionary.txt"
        filepath.write_bytes(response.content)

        print("Jeff Sackmann ATP matches dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading Jeff Sackmann ATP matches dataset: {e}")


def load_dataset(year_count: int = 10) -> pd.DataFrame:
    most_recent_years = sorted(OUTPUT_DIR.glob("atp_matches_*.csv"), reverse=True)[
        :year_count
    ]
    dfs = [pd.read_csv(filepath) for filepath in most_recent_years]

    return pd.concat(dfs, ignore_index=True)
