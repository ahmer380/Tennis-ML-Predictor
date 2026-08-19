from fastapi import APIRouter

from backend.schemas import PlayerProfileRequest, PlayerProfileResponse

from ml.src.pipelines.predict import build_player_profile

router = APIRouter(prefix="/profile", tags=["Player Profile"])


@router.post("", summary="Get the profile of a tennis player", response_model=PlayerProfileResponse)
def get_player_profile(request: PlayerProfileRequest):
    profile = build_player_profile(player_name=request.player_name, player_year=request.player_year)

    return profile.to_dict()
