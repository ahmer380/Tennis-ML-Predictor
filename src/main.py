from src.step_1_load_dataset import download_dataset, load_dataset
from src.step_2_preprocess_data import preprocess_matches, audit_dataset
from src.step_3_feature_engineering import (
    engineer_features,
    audit_player_states,
    audit_player_h2h,
    audit_player_tournament_run,
)
from src.step_4_split_dataset import split_dataset
from src.models.neural_network import learn

if __name__ == "__main__":
    print("Downloading dataset...\n")
    download_dataset()
    df = load_dataset()

    print("\nPreprocessing dataset...\n")
    df_preprocessed = preprocess_matches(df)
    # audit_dataset(df_preprocessed)

    print("\nEngineering features...\n")
    df_features, player_states = engineer_features(df_preprocessed)
    # audit_dataset(df_features)
    # audit_player_states(player_states)
    # audit_player_h2h(df_features, "Novak Djokovic", "Carlos Alcaraz")
    # audit_player_tournament_run(df_features, "Rafael Nadal", "US Open", 2019)

    print("\nSplitting dataset...\n")
    X_train, y_train, X_validation, y_validation, X_test, y_test = split_dataset(df_features)
    # audit_dataset(X_train.assign(player_A_win=y_train))
    # audit_dataset(X_validation.assign(player_A_win=y_validation))
    # audit_dataset(X_test.assign(player_A_win=y_test))

    print("\nTraining neural network...\n")
    model = learn(X_train, y_train)
    print("Finished training.")

    # TODO: Step 5: Train and evaluate machine learning models
    # 5.a Baseline model (choosing higher elo)
    # 5.b Neural network
    # 5.c Logistic regression
    # 5.d Random forest
    # 5.e XGBoost
    # just a and b for now, will add c, d, and e in the future

    # TODO: Step 6: Evaluate model performance and visualise results
