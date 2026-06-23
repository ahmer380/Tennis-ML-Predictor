from datetime import datetime
from fastapi import APIRouter

from src.pipelines.predict import predict

from fastapi import APIRouter

from src.pipelines.predict import predict

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.get("")
def predict_match(
    player_a_name: str,
    player_b_name: str,
    surface: str = "Hard",
    best_of: int = 3,
    model: str = "xgboost",
    player_a_year: int = datetime.now().year,
    player_b_year: int = datetime.now().year,
):
    player_a, player_b, player_a_win_probability = predict(
        model_type=model,
        player_a_name=player_a_name,
        player_b_name=player_b_name,
        surface=surface,
        best_of=best_of,
        player_a_year=player_a_year,
        player_b_year=player_b_year,
    )

    return {
        "player_a": player_a.to_dict(),
        "player_b": player_b.to_dict(),
        "player_a_win_probability": round(player_a_win_probability, 4),
    }
