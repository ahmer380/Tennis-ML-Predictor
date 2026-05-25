import argparse
import shutil
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import kagglehub
import requests


class DataLoader(ABC):
    def __init__(self, data_root: Path = Path(__file__).resolve().parent.parent / "data") -> None:
        self.output_dir = data_root / self.subfolder_name

    @property
    @abstractmethod
    def subfolder_name(self) -> str:
        pass

    @abstractmethod
    def download_dataset(self) -> None:
        pass


class JeffSackmannDataLoader(DataLoader):
    BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

    @property
    def subfolder_name(self) -> str:
        return "jeff_sackmann"

    def download_dataset(self, year_count: int = 10) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for year in range(date.today().year - year_count, date.today().year + 1):
                url = f"{self.BASE_URL}/atp_matches_{year}.csv"
                response = requests.get(url)
                filepath = self.output_dir / f"atp_matches_{year}.csv"
                filepath.write_bytes(response.content)
            
            print("Jeff Sackmann ATP matches dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading Jeff Sackmann ATP matches dataset: {e}")

class KagglehubDataLoader(DataLoader):
    BASE_URL = "dissfya/atp-tennis-2000-2023daily-pull"
    KAGGLE_FILE = "atp_tennis.csv"

    @property
    def subfolder_name(self) -> str:
        return "kagglehub"

    def download_dataset(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try: 
            filepath = self.output_dir / f"live_dataset_{date.today().isoformat()}.csv"
            path = kagglehub.dataset_download(self.BASE_URL, path=self.KAGGLE_FILE)
            shutil.move(path, filepath)
            print("Kagglehub ATP matches dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading Kagglehub ATP matches dataset: {e}")


LOADERS: dict[str, type[DataLoader]] = {
    "jeff_sackmann": JeffSackmannDataLoader,
    "kagglehub": KagglehubDataLoader,
}


def main():
    parser = argparse.ArgumentParser(description="Download tennis match datasets.")
    parser.add_argument(
        "--source",
        choices=["jeff_sackmann", "kagglehub", "all"],
        default="all",
        help="Which dataset source to download (default: all)",
    )
    args = parser.parse_args()

    sources: list[str] = list(LOADERS.keys()) if args.source == "all" else [args.source]

    for source in sources:
        loader = LOADERS[source]()
        loader.download_dataset()


if __name__ == "__main__":
    main()
