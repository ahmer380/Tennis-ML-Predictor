from fastapi import APIRouter

from backend.schemas import PlayerProfileRequest, PlayerProfileResponse

from ml.src.pipelines.predict import build_player_profile

router = APIRouter(prefix="/profile", tags=["Player Profile"])


@router.post("", summary="Get the profile of a tennis player", response_model=PlayerProfileResponse)
def get_player_profile(request: PlayerProfileRequest):
    profile_dict = build_player_profile(player_name=request.player_name, player_year=request.player_year).to_dict()

    profile_dict["bio"] = "Temporary biography of the player."
    profile_dict["image_url"] = f"/static/player_images/{request.player_name.lower().replace(' ', '_')}.jpg"

    return profile_dict
