from pydantic import BaseModel


class PlayerProfileResponse(BaseModel):
    name: str
    rank: int
    rank_points: int
    age: float
    height: int
    global_elo: float
    hard_elo: float
    clay_elo: float
    grass_elo: float


class PredictionResponse(BaseModel):
    player_a: PlayerProfileResponse
    player_b: PlayerProfileResponse
    player_a_win_probability: float
