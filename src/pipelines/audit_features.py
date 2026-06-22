from src.data.download_dataset import download_dataset
from src.data.load_dataset import load_dataset
from src.data.preprocess_dataset import preprocess_dataset

from src.feature.feature_engineering import FeatureEngineer

from src.utils.audit import (
    audit_dataset,
    audit_player_profiles,
    audit_player_h2h,
    audit_match,
    audit_player_tournament_run,
)
from src.utils.plot import plot_player_career_elo_trajectory


def audit_features():
    """Pipeline to audit the engineered features."""

    # Download, load, and preprocess the dataset
    download_dataset()
    df = load_dataset()
    df_preprocessed = preprocess_dataset(df)

    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_dataframe(df_preprocessed)
    player_profiles = feature_engineer.player_profiles

    audit_dataset(df_features)
    audit_player_profiles(player_profiles)
    audit_player_h2h(df_features, "Novak Djokovic", "Carlos Alcaraz")
    audit_match(df_features, "Novak Djokovic", "Rafael Nadal", "Rome Masters", 2011)
    audit_player_tournament_run(df_features, "Rafael Nadal", "US Open", 2019)
    plot_player_career_elo_trajectory(df_features, "Rafael Nadal")


if __name__ == "__main__":
    audit_features()
