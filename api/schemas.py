from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

CURRENT_YEAR = datetime.now().year


class PlayerProfileRequest(BaseModel):
    player_name: str = Field(..., description="Full name of the player", example="Jannik Sinner")
    player_year: int = Field(
        CURRENT_YEAR, ge=2010, le=CURRENT_YEAR, description="(end of) Cutoff year for the player's stats"
    )


class PlayerProfileResponse(BaseModel):
    name: str = Field(..., description="Player full name", example="Jannik Sinner")
    rank: int = Field(..., description="ATP ranking position", example=1)
    rank_points: int = Field(..., description="ATP ranking points total", example=14350)
    age: float = Field(..., description="Player age in years", example=24)
    height: int = Field(..., description="Player height in cm", example=191)
    global_elo: float = Field(..., description="Overall Elo rating across all surfaces", example=1929.19)
    hard_elo: float = Field(..., description="Elo rating on hard courts", example=1901.63)
    clay_elo: float = Field(..., description="Elo rating on clay courts", example=1712.91)
    grass_elo: float = Field(..., description="Elo rating on grass courts", example=1649.43)


class PredictionRequest(BaseModel):
    player_a_name: str = Field(..., description="Full name of player A", example="Jannik Sinner")
    player_b_name: str = Field(..., description="Full name of player B", example="Carlos Alcaraz")
    surface: Literal["Hard", "Clay", "Grass"] = Field(..., description="Surface of the court")
    best_of: Literal[3, 5] = Field(..., description="Best-of value for the match")
    model: Literal["elo", "mlp", "xgboost"] = Field("xgboost", description="Model type to use for prediction")
    player_a_year: int = Field(
        CURRENT_YEAR, ge=2010, le=CURRENT_YEAR, description="(end of) Cutoff year for Player A's stats"
    )
    player_b_year: int = Field(
        CURRENT_YEAR, ge=2010, le=CURRENT_YEAR, description="(end of) Cutoff year for Player B's stats"
    )


class PredictionResponse(BaseModel):
    player_a: PlayerProfileResponse = Field(
        ...,
        description="Profile of Player A",
        example=PlayerProfileResponse(
            name="Jannik Sinner",
            rank=1,
            rank_points=14350,
            age=24,
            height=191,
            global_elo=1929.19,
            hard_elo=1901.63,
            clay_elo=1712.91,
            grass_elo=1649.43,
        ),
    )
    player_b: PlayerProfileResponse = Field(
        ...,
        description="Profile of Player B",
        example=PlayerProfileResponse(
            name="Carlos Alcaraz",
            rank=2,
            rank_points=13240,
            age=22,
            height=183,
            global_elo=1807.11,
            hard_elo=1756.92,
            clay_elo=1755.27,
            grass_elo=1721.29,
        ),
    )
    player_a_win_probability: float = Field(
        ..., description="Predicted probability that Player A wins the match", example=0.724
    )
