import argparse

from src.data.download_dataset import download_dataset
from src.data.load_dataset import load_dataset
from src.data.preprocess_dataset import preprocess_dataset

from src.feature.feature_engineering import FeatureEngineer, get_player_profile_by_name
from src.feature.prepare_ml_dataset import prepare_ml_dataset

from src.models.elo import TennisPredictorElo
from src.models.mlp import TennisPredictorMLP
from src.models.xgboost import TennisPredictorXGBoost

from src.evaluate.evaluate_model import evaluate_model, predict_match


def train(model_type: str):
    """Pipeline to train a tennis match predictor model."""

    # Download, load, and preprocess the dataset
    download_dataset()
    df = load_dataset()
    df_preprocessed = preprocess_dataset(df)

    # Engineer features and prepare the dataset for machine learning
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_dataframe(df_preprocessed)
    player_profiles = feature_engineer.player_profiles
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_ml_dataset(df_features)

    # Train the model based on the specified model type
    if model_type == "elo":
        model = TennisPredictorElo()
    elif model_type == "mlp":
        model = TennisPredictorMLP()
    elif model_type == "xgboost":
        model = TennisPredictorXGBoost()
    model.learn(X_train, y_train, X_validation, y_validation)
    model.save()

    # Evaluate the model on the test set and save evaluation metrics and plots
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
