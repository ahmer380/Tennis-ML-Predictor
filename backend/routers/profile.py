from pathlib import Path

from fastapi import APIRouter

from backend.schemas import PlayerProfileRequest, PlayerProfileResponse

from ml.src.pipelines.predict import build_player_profile

IMAGE_DIR = Path(__file__).resolve().parents[1] / "static" / "player_images"

router = APIRouter(prefix="/profile", tags=["Player Profile"])


@router.post("", summary="Get the profile of a tennis player", response_model=PlayerProfileResponse)
def get_player_profile(request: PlayerProfileRequest):
    profile_dict = build_player_profile(player_name=request.player_name, player_year=request.player_year).to_dict()

    profile_dict["bio"] = "Temporary biography of the player."

    player_images = list(IMAGE_DIR.glob("*.jpg"))
    profile_dict["image_url"] = "/static/player_images/default.jpg"
    for image in player_images:
        if request.player_name.lower().replace(" ", "_") == image.stem:
            profile_dict["image_url"] = f"/static/player_images/{image.name}"
            break

    return profile_dict
