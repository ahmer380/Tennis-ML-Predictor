from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.match_record import (
    JEFF_SACKMANN_COLUMNS,
    MatchRecord,
    RowAuditResult,
    audit_jeff_sackmann_rows,
    from_jeff_sackmann_row,
)


class JeffSackmannDataLoader:
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
    BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

    @classmethod
    def download_dataset(cls, year_count: int = 10) -> None:
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        try:
            for year in range(date.today().year - year_count, date.today().year + 1):
                url = f"{cls.BASE_URL}/atp_matches_{year}.csv"
                response = requests.get(url)
                filepath = cls.OUTPUT_DIR / f"atp_matches_{year}.csv"
                filepath.write_bytes(response.content)

            # download schema file for information about the columns
            url = f"{cls.BASE_URL}/matches_data_dictionary.txt"
            response = requests.get(url)
            filepath = cls.OUTPUT_DIR / "matches_data_dictionary.txt"
            filepath.write_bytes(response.content)

            print("Jeff Sackmann ATP matches dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading Jeff Sackmann ATP matches dataset: {e}")

    @classmethod
    def load_dataset(cls, year_count: int = 10) -> list[MatchRecord]:
        records: list[MatchRecord] = []
        sorted_files = sorted(cls.OUTPUT_DIR.glob("atp_matches_*.csv"), reverse=True)
        for filepath in sorted_files[:year_count]:
            df = pd.read_csv(filepath)
            if list(df.columns) != JEFF_SACKMANN_COLUMNS:
                raise ValueError(
                    f"Unexpected columns in {filepath.name}: expected {len(JEFF_SACKMANN_COLUMNS)}, "
                    f"got {len(df.columns)}"
                )

            for _, row in df.iterrows():
                records.append(from_jeff_sackmann_row(row))

        return records

    @classmethod
    def audit_dataset(cls) -> RowAuditResult:
        return audit_jeff_sackmann_rows(cls.OUTPUT_DIR)


if __name__ == "__main__":
    JeffSackmannDataLoader.download_dataset(year_count=10)
    # records = JeffSackmannDataLoader.load_dataset(year_count=10)
