from src.step_1_load_dataset import download_dataset, load_dataset

if __name__ == "__main__":
    # download the dataset if not already downloaded
    download_dataset()

    # load the dataset into a pandas DataFrame
    df = load_dataset(year_count=1)

    print(df.head())
