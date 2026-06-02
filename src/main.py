from src.step_1_load_dataset import download_dataset, load_dataset
from src.step_2_preprocess_data import preprocess_matches, audit_dataset
from src.step_3_feature_engineering import engineer_features, audit_elo_ratings

if __name__ == "__main__":
    print("Downloading dataset...\n")
    download_dataset()
    df = load_dataset()

    print("\nPreprocessing dataset...\n")
    df_preprocessed = preprocess_matches(df)
    audit_dataset(df_preprocessed)

    print("\nEngineering features...\n")
    df_features, elo_ratings = engineer_features(df_preprocessed)
    audit_elo_ratings(elo_ratings)

    # print(df_features[(df_features["tourney_id"] == "2026-580") & (df_features["round"] == "F")].iloc[-1])
