import argparse

from src.models.elo import TennisPredictorElo
from src.models.mlp import TennisPredictorMLP
from src.models.xgboost import TennisPredictorXGBoost

from src.data.download_dataset import download_dataset
from src.data.load_dataset import load_dataset
from src.data.preprocess_dataset import preprocess_dataset

from src.feature.prepare_ml_dataset import prepare_ml_dataset

from src.step_3_feature_engineering import (
    engineer_features,
    get_player_profile_by_name,
)
from src.step_5_evaluate_model import evaluate_model, predict_match

from src.utils.audit import (
    audit_dataset,
    audit_player_profiles,
    audit_player_h2h,
    audit_match,
    audit_player_tournament_run,
)
from src.utils.plot import plot_player_career_elo_trajectory


def train(model_type: str):
    # Download, load, and preprocess the dataset
    download_dataset()
    df = load_dataset()
    df_preprocessed = preprocess_dataset(df)
    # audit_dataset(df_preprocessed)

    print("\nEngineering features...")
    df_features, player_profiles = engineer_features(df_preprocessed)
    # audit_dataset(df_features)
    # audit_player_profiles(player_profiles)
    # audit_player_h2h(df_features, "Novak Djokovic", "Carlos Alcaraz")
    # audit_match(df_features, "Novak Djokovic", "Rafael Nadal", "Rome Masters", 2011)
    # audit_player_tournament_run(df_features, "Rafael Nadal", "US Open", 2019)
    # plot_player_career_elo_trajectory(df_features, "Rafael Nadal")

    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_ml_dataset(df_features)
    # audit_dataset(X_train.assign(player_A_win=y_train))
    # audit_dataset(X_validation.assign(player_A_win=y_validation))
    # audit_dataset(X_test.assign(player_A_win=y_test))

    print("\nTraining model...\n")
    if model_type == "elo":
        model = TennisPredictorElo()
    elif model_type == "mlp":
        model = TennisPredictorMLP()
    elif model_type == "xgboost":
        model = TennisPredictorXGBoost()
    model.learn(X_train, y_train, X_validation, y_validation)
    model.save()

    print("\nEvaluating model...\n")
    evaluate_model(model, X_test, y_test, save_data=True)
    predict_match(
        model=model,
        player_a_profile=get_player_profile_by_name(player_profiles, "Novak Djokovic"),
        player_b_profile=get_player_profile_by_name(player_profiles, "Carlos Alcaraz"),
        surface="Clay",
        best_of=5,
    )
    predict_match(
        model=model,
        player_a_profile=get_player_profile_by_name(player_profiles, "Carlos Alcaraz"),
        player_b_profile=get_player_profile_by_name(player_profiles, "Jannik Sinner"),
        surface="Hard",
        best_of=3,
    )
    predict_match(
        model=model,
        player_a_profile=get_player_profile_by_name(player_profiles, "Jannik Sinner"),
        player_b_profile=get_player_profile_by_name(player_profiles, "Kei Nishikori"),
        surface="Grass",
        best_of=3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tennis match predictor model.")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["elo", "mlp", "xgboost"],
        help="The model to train (default: xgboost)",
    )
    args = parser.parse_args()

    train(model_type=args.model)
