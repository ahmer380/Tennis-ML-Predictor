from datetime import datetime
from fastapi import APIRouter

from src.pipelines.predict import build_player_profile

router = APIRouter(prefix="/profile", tags=["Player Profile"])


@router.get("")
def get_player_profile(
    player_name: str,
    player_year: int = datetime.now().year,
):
    profile = build_player_profile(player_name, player_year)

    return profile.to_dict()
