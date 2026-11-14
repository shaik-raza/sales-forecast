from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from api.forecast.routes import router as forecast_router
from api.auth.routes import router as auth_router

app = FastAPI(
    title="Sales Forecast API",
    swagger_ui_parameters={"persistAuthorization": True},
)
app.include_router(forecast_router)
app.include_router(auth_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

Instrumentator().instrument(app).expose(app)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")