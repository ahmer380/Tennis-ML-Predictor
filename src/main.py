from src.step_1_load_dataset import download_dataset, load_dataset
from src.step_2_preprocess_data import preprocess_matches, audit_dataset

if __name__ == "__main__":
    download_dataset()
    df = load_dataset()
    audit_dataset(df)

    print("\nPreprocessing dataset...\n")

    df_preprocessed = preprocess_matches(df)
    audit_dataset(df_preprocessed)
