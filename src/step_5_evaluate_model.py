from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.model import TennisPredictorModel

from src.feature.features import FINALISED_ML_FEATURES

from src.step_3_feature_engineering import PlayerProfile


def plot_confusion_matrix(model: TennisPredictorModel, y_true: pd.Series, y_pred: np.ndarray) -> None:
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    confusion_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Player 0", "Player 1"])

    fig, ax = plt.subplots(figsize=(5, 4))
    confusion_display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model.instance_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(model.instance_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_histogram(model: TennisPredictorModel, y_prob: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob, bins=20, range=(0.0, 1.0), color="#4C72B0", edgecolor="white")
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{model.instance_name} Prediction Histogram")
    fig.tight_layout()
    fig.savefig(model.instance_dir / "prediction_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_metrics(
    model: TennisPredictorModel, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray
) -> None:
    metrics = {
        "Log Loss": log_loss(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }

    with open(model.instance_dir / "evaluation_metrics.txt", "w") as f:
        f.write(f"Evaluation Metrics for {model.instance_name}\n\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {format_metric(metric_value)}\n")


def format_metric(value: Any) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def evaluate_model(
    model: TennisPredictorModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_data: bool = True,
) -> None:
    y_prob = model.predict(X_test)
    y_pred = model.predict_class(X_test)

    if save_data:
        plot_confusion_matrix(model, y_test, y_pred)
        plot_prediction_histogram(model, y_prob)
        save_evaluation_metrics(model, y_test, y_pred, y_prob)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Model:         {model.instance_name}")
    print()
    print(f"Log Loss:      {format_metric(log_loss(y_test, y_prob))}")
    print(f"Brier Score:   {format_metric(brier_score_loss(y_test, y_prob))}")
    print(f"ROC AUC:       {format_metric(roc_auc_score(y_test, y_prob))}")
    print()
    print(f"Accuracy:      {format_metric(accuracy_score(y_test, y_pred))}")
    print(f"Precision:     {format_metric(precision_score(y_test, y_pred, zero_division=0))}")
    print(f"Recall:        {format_metric(recall_score(y_test, y_pred, zero_division=0))}")
    print(f"F1 Score:      {format_metric(f1_score(y_test, y_pred, zero_division=0))}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    if save_data:
        print("Evaluation plots saved to:")
        print(f"{model.instance_dir}/confusion_matrix.png")
        print(f"{model.instance_dir}/prediction_histogram.png")
        print(f"{model.instance_dir}/evaluation_metrics.txt")
    print()
    print("=" * 50)


def predict_match(
    model: TennisPredictorModel,
    player_a_profile: PlayerProfile,
    player_b_profile: PlayerProfile,
    surface: str,
    best_of: int,
) -> float:
    """Predict the probability of Player A winning against Player B."""
    win_probability = (
        _predict(model, player_a_profile, player_b_profile, surface, best_of)
        + (1 - _predict(model, player_b_profile, player_a_profile, surface, best_of))
    ) / 2  # Average the predictions across both directions (A vs B and B vs A) to account for potential asymmetry in the model

    print("=" * 50)
    print(f"Win probability according to {model.instance_name}:")
    print(f"{player_a_profile.name}: {win_probability:.2%}")
    print(f"{player_b_profile.name}: {1 - win_probability:.2%}")
    print("=" * 50)
    print()

    return win_probability


def _predict(
    model: TennisPredictorModel,
    player_a_profile: PlayerProfile,
    player_b_profile: PlayerProfile,
    surface: str,
    best_of: int,
) -> float:
    """Private function to predict the probability of Player A winning against Player B in a single direction (A vs B)."""

    feature_vector = {
        # Ranking
        "player_A_rank": player_a_profile.rank,
        "player_B_rank": player_b_profile.rank,
        "rank_diff": player_a_profile.rank - player_b_profile.rank,
        "player_A_rank_points": player_a_profile.rank_points,
        "player_B_rank_points": player_b_profile.rank_points,
        "rank_points_diff": player_a_profile.rank_points - player_b_profile.rank_points,
        # Experience
        "player_A_global_matches_played": player_a_profile.get_matches_played("global"),
        "player_B_global_matches_played": player_b_profile.get_matches_played("global"),
        "global_matches_played_diff": player_a_profile.get_matches_played("global")
        - player_b_profile.get_matches_played("global"),
        "player_A_surface_matches_played": player_a_profile.get_matches_played(surface),
        "player_B_surface_matches_played": player_b_profile.get_matches_played(surface),
        "surface_matches_played_diff": player_a_profile.get_matches_played(surface)
        - player_b_profile.get_matches_played(surface),
        "player_A_global_matches_played_last_365": player_a_profile.get_matches_played(
            "global", pd.Timestamp.now() - pd.DateOffset(years=1)
        ),
        "player_B_global_matches_played_last_365": player_b_profile.get_matches_played(
            "global", pd.Timestamp.now() - pd.DateOffset(years=1)
        ),
        "global_matches_played_last_365_diff": player_a_profile.get_matches_played(
            "global", pd.Timestamp.now() - pd.DateOffset(years=1)
        )
        - player_b_profile.get_matches_played("global", pd.Timestamp.now() - pd.DateOffset(years=1)),
        # Elo
        "player_A_global_elo": player_a_profile.elos["global"],
        "player_B_global_elo": player_b_profile.elos["global"],
        "global_elo_diff": player_a_profile.elos["global"] - player_b_profile.elos["global"],
        "player_A_surface_elo": player_a_profile.elos[surface],
        "player_B_surface_elo": player_b_profile.elos[surface],
        "surface_elo_diff": player_a_profile.elos[surface] - player_b_profile.elos[surface],
        # Form
        "player_A_global_win_pct_last_10": player_a_profile.get_recent_win_percentage(10, "global"),
        "player_B_global_win_pct_last_10": player_b_profile.get_recent_win_percentage(10, "global"),
        "global_win_pct_last_10_diff": player_a_profile.get_recent_win_percentage(10, "global")
        - player_b_profile.get_recent_win_percentage(10, "global"),
        "player_A_global_win_pct_last_25": player_a_profile.get_recent_win_percentage(25, "global"),
        "player_B_global_win_pct_last_25": player_b_profile.get_recent_win_percentage(25, "global"),
        "global_win_pct_last_25_diff": player_a_profile.get_recent_win_percentage(25, "global")
        - player_b_profile.get_recent_win_percentage(25, "global"),
        "player_A_global_win_pct_last_50": player_a_profile.get_recent_win_percentage(50, "global"),
        "player_B_global_win_pct_last_50": player_b_profile.get_recent_win_percentage(50, "global"),
        "global_win_pct_last_50_diff": player_a_profile.get_recent_win_percentage(50, "global")
        - player_b_profile.get_recent_win_percentage(50, "global"),
        "player_A_global_win_pct_last_100": player_a_profile.get_recent_win_percentage(100, "global"),
        "player_B_global_win_pct_last_100": player_b_profile.get_recent_win_percentage(100, "global"),
        "global_win_pct_last_100_diff": player_a_profile.get_recent_win_percentage(100, "global")
        - player_b_profile.get_recent_win_percentage(100, "global"),
        "player_A_surface_win_pct_last_100": player_a_profile.get_recent_win_percentage(100, surface),
        "player_B_surface_win_pct_last_100": player_b_profile.get_recent_win_percentage(100, surface),
        "surface_win_pct_last_100_diff": player_a_profile.get_recent_win_percentage(100, surface)
        - player_b_profile.get_recent_win_percentage(100, surface),
        # Head-to-head
        "player_A_h2h_wins": player_a_profile.get_h2h_wins(player_b_profile.id),
        "player_B_h2h_wins": player_b_profile.get_h2h_wins(player_a_profile.id),
        "h2h_diff": player_a_profile.get_h2h_wins(player_b_profile.id)
        - player_b_profile.get_h2h_wins(player_a_profile.id),
        # Game stats
        "player_A_ace_pct": player_a_profile.get_recent_game_stat_average("p_ace", 100, 0.08),
        "player_B_ace_pct": player_b_profile.get_recent_game_stat_average("p_ace", 100, 0.08),
        "ace_pct_diff": player_a_profile.get_recent_game_stat_average("p_ace", 100, 0.08)
        - player_b_profile.get_recent_game_stat_average("p_ace", 100, 0.08),
        "player_A_df_pct": player_a_profile.get_recent_game_stat_average("p_df", 100, 0.05),
        "player_B_df_pct": player_b_profile.get_recent_game_stat_average("p_df", 100, 0.05),
        "df_pct_diff": player_a_profile.get_recent_game_stat_average("p_df", 100, 0.05)
        - player_b_profile.get_recent_game_stat_average("p_df", 100, 0.05),
        "player_A_1st_in_pct": player_a_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5),
        "player_B_1st_in_pct": player_b_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5),
        "1st_in_pct_diff": player_a_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_1st_in", 100, 0.5),
        "player_A_1st_won_pct": player_a_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5),
        "player_B_1st_won_pct": player_b_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5),
        "1st_won_pct_diff": player_a_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_1st_won", 100, 0.5),
        "player_A_2nd_won_pct": player_a_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5),
        "player_B_2nd_won_pct": player_b_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5),
        "2nd_won_pct_diff": player_a_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_2nd_won", 100, 0.5),
        "player_A_bp_saved_pct": player_a_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5),
        "player_B_bp_saved_pct": player_b_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5),
        "bp_saved_pct_diff": player_a_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_bp_saved", 100, 0.5),
        "player_A_rp_won_pct": player_a_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5),
        "player_B_rp_won_pct": player_b_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5),
        "rp_won_pct_diff": player_a_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_rp_won", 100, 0.5),
        "player_A_bp_won_pct": player_a_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5),
        "player_B_bp_won_pct": player_b_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5),
        "bp_won_pct_diff": player_a_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5)
        - player_b_profile.get_recent_game_stat_average("p_bp_won", 100, 0.5),
        # Physical
        "player_A_age": player_a_profile.age,
        "player_B_age": player_b_profile.age,
        "age_diff": player_a_profile.age - player_b_profile.age,
        "player_A_ht": player_a_profile.ht,
        "player_B_ht": player_b_profile.ht,
        "ht_diff": player_a_profile.ht - player_b_profile.ht,
        # Tournament fatigue
        "player_A_tournament_minutes": 0,
        "player_B_tournament_minutes": 0,
        "tournament_minutes_diff": 0,
        # Match context
        "best_of_5": 1 if best_of == 5 else 0,
        "hard_surface": 1 if surface == "Hard" else 0,
        "clay_surface": 1 if surface == "Clay" else 0,
        "grass_surface": 1 if surface == "Grass" else 0,
    }

    feature_vector = {k: v for k, v in feature_vector.items() if k in FINALISED_ML_FEATURES}
    assert set(feature_vector.keys()) == set(FINALISED_ML_FEATURES) - {
        "player_A_win"
    }, "Feature vector keys do not match expected features"

    win_probability = model.predict(pd.DataFrame([feature_vector]))[0]
    return win_probability
