from fastapi import FastAPI

from api.routers.prediction import router as prediction_router
from api.routers.profile import router as profile_router

app = FastAPI(
    title="Tennis Predictor API",
    version="1.0.0",
)

app.include_router(prediction_router)
app.include_router(profile_router)
