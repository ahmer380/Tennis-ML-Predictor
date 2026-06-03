from src.step_1_load_dataset import download_dataset, load_dataset
from src.step_2_preprocess_data import preprocess_matches, audit_dataset
from src.step_3_feature_engineering import (
    engineer_features,
    audit_player_states,
    audit_player_h2h,
    audit_player_tournament_run,
)

if __name__ == "__main__":
    print("Downloading dataset...\n")
    download_dataset()
    df = load_dataset()

    print("\nPreprocessing dataset...\n")
    df_preprocessed = preprocess_matches(df)
    # audit_dataset(df_preprocessed)

    print("\nEngineering features...\n")
    df_features, player_states = engineer_features(df_preprocessed)
    # audit_player_states(player_states)
    # audit_player_h2h(df_features, "Novak Djokovic", "Carlos Alcaraz")
    # audit_player_tournament_run(df_features, "Rafael Nadal", "US Open", 2019)

    # TODO (step 4)
    print(df_features)
