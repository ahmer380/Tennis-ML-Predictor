from src.models.elo import TennisPredictorElo
from src.models.mlp import TennisPredictorMLP

from src.step_1_load_dataset import download_dataset, load_dataset
from src.step_2_preprocess_data import preprocess_matches, audit_dataset
from src.step_3_feature_engineering import (
    engineer_features,
    audit_player_states,
    audit_player_h2h,
    audit_match,
    audit_player_tournament_run,
    plot_player_career_elo_trajectory,
)
from src.step_4_split_dataset import split_dataset
from src.step_5_evaluate_model import evaluate_model

if __name__ == "__main__":
    print("Downloading dataset...\n")
    # download_dataset()
    df = load_dataset()

    print("\nPreprocessing dataset...\n")
    df_preprocessed = preprocess_matches(df)
    # audit_dataset(df_preprocessed)

    print("\nEngineering features...\n")
    df_features, player_states = engineer_features(df_preprocessed)
    # audit_dataset(df_features)
    # audit_player_states(player_states)
    # audit_player_h2h(df_features, "Novak Djokovic", "Carlos Alcaraz")
    # audit_match(df_features, "Novak Djokovic", "Rafael Nadal", "Rome Masters", 2011)
    # audit_player_tournament_run(df_features, "Rafael Nadal", "US Open", 2019)
    # plot_player_career_elo_trajectory(df_features, "Rafael Nadal")

    print("\nSplitting dataset...\n")
    X_train, y_train, X_validation, y_validation, X_test, y_test = split_dataset(df_features)
    # audit_dataset(X_train.assign(player_A_win=y_train))
    # audit_dataset(X_validation.assign(player_A_win=y_validation))
    # audit_dataset(X_test.assign(player_A_win=y_test))

    # print("\nTraining neural network...\n")
    # model = TennisPredictorMLP()
    # model.learn(X_train, y_train, X_validation, y_validation)
    # model.save()
    # print("Finished training.")

    model = TennisPredictorMLP.load(version=1)

    print("\nEvaluating model...\n")
    evaluate_model(model, X_test, y_test, save_plots=False)

    # TODO: Expand test set to evaluate tournament run predictions
    # TODO: Train XGBoost model and compare with neural network and baseline model
    # TODO: Maybe add sns pairplot to visualize features and their relationships with the target variable?
