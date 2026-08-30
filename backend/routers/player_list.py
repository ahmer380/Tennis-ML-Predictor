from datetime import datetime

from fastapi import APIRouter

from backend.schemas import PlayerProfileRequest, PlayerListResponse
from backend.routers.profile import get_player_profile

CURRENT_YEAR = datetime.now().year
PLAYERS = [
    "Roger Federer",
    "Rafael Nadal",
    "Novak Djokovic",
    "Carlos Alcaraz",
    "Jannik Sinner",
]

router = APIRouter(prefix="/players", tags=["Player List"])


@router.get("", summary="Get a list of all tennis players", response_model=PlayerListResponse)
def get_player_list():
    player_profiles = []
    for player_name in PLAYERS:
        profile_request = PlayerProfileRequest(**{"player_name": player_name, "player_year": CURRENT_YEAR})
        profile_response = get_player_profile(profile_request)
        player_profiles.append(profile_response)

    return {"players": player_profiles}
