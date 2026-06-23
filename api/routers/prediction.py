from fastapi import APIRouter

from api.schemas import PredictionRequest, PredictionResponse

from src.pipelines.predict import predict

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse)
def predict_match(request: PredictionRequest):
    player_a, player_b, player_a_win_probability = predict(
        model_type=request.model,
        player_a_name=request.player_a_name,
        player_b_name=request.player_b_name,
        surface=request.surface,
        best_of=request.best_of,
        player_a_year=request.player_a_year,
        player_b_year=request.player_b_year,
    )

    return {
        "player_a": player_a.to_dict(),
        "player_b": player_b.to_dict(),
        "player_a_win_probability": round(player_a_win_probability, 4),
    }
