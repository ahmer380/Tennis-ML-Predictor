from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.routers.prediction import router as prediction_router
from backend.routers.profile import router as profile_router
from backend.routers.player_list import router as player_list_router

app = FastAPI(
    title="Tennis Predictor API",
    version="1.0.0",
)

app.include_router(prediction_router)
app.include_router(profile_router)
app.include_router(player_list_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=str(exc),
    )
