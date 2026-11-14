from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.auth.jwt_handler import get_current_user
from api.db.database import Base, engine, get_db
from api.db.models import ForecastHistory
from api.forecast.schemas import (
    CompareRequest,
    ForecastRequest,
    ForecastResponse,
    ModelComparisonResponse,
)
from ml.features import FEATURE_COLUMNS, build_features

Base.metadata.create_all(bind=engine)

router = APIRouter()
ARTIFACT_DIR = Path("ml/artifacts")
DATA_PATH = Path("Data/processed/train_merged_clean.csv")
MODEL_NAMES = ("xgboost", "lightgbm", "catboost", "random_forest")


@lru_cache(maxsize=1)
def _load_models() -> tuple[dict[str, Any], dict[str, float]]:
    """Load model artifacts once, on the first forecast request."""
    models = {
        name: joblib.load(ARTIFACT_DIR / f"{name}.joblib")["model"]
        for name in MODEL_NAMES
    }
    ensemble = joblib.load(ARTIFACT_DIR / "ensemble.joblib")
    return models, ensemble["weights"]


@lru_cache(maxsize=1)
def _load_metrics() -> dict[str, Any]:
    """Load metrics only when the metrics endpoint is requested."""
    with (ARTIFACT_DIR / "model_metrics.json").open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@lru_cache(maxsize=1)
def _load_store_details() -> pd.DataFrame:
    columns = [
        "Store",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]
    data = pd.read_csv(DATA_PATH, usecols=columns)
    return data.drop_duplicates("Store").set_index("Store")


@router.get("/stores/{store_id}")
def get_store_details(store_id: int):
    details = _load_store_details()
    if store_id not in details.index:
        raise HTTPException(status_code=404, detail=f"Store {store_id} was not found")
    values = details.loc[store_id].where(pd.notna(details.loc[store_id]), 0)
    return {
        "Store": store_id,
        "StoreType": int(values["StoreType"]),
        "Assortment": int(values["Assortment"]),
        "CompetitionDistance": float(values["CompetitionDistance"]),
        "CompetitionOpenSinceMonth": float(values["CompetitionOpenSinceMonth"]),
        "CompetitionOpenSinceYear": float(values["CompetitionOpenSinceYear"]),
        "Promo2": int(values["Promo2"]),
        "Promo2SinceWeek": float(values["Promo2SinceWeek"]),
        "Promo2SinceYear": float(values["Promo2SinceYear"]),
    }


def _historical_store_rows(store: int, date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "Store",
        "DayOfWeek",
        "Date",
        "Sales",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
        "Year",
        "Month",
        "Day",
        "WeekOfYear",
    ]
    data = pd.read_csv(DATA_PATH, usecols=columns)
    history = data[(data["Store"] == store) & (pd.to_datetime(data["Date"]) < date)]
    if history.empty:
        raise HTTPException(status_code=400, detail=f"No historical data found for store {store}")
    return history


def _request_row(request_data: dict[str, Any], date: pd.Timestamp) -> pd.DataFrame:
    row = request_data.copy()
    row.update(
        {
            "Date": date,
            "Sales": 0.0,
            "Year": date.year,
            "Month": date.month,
            "Day": date.day,
            "DayOfWeek": date.dayofweek + 1,
            "WeekOfYear": int(date.isocalendar().week),
        }
    )
    row.setdefault("PromoInterval", "None")
    return pd.DataFrame([row])


def _features_for_request(request_data: dict[str, Any], date: pd.Timestamp) -> pd.DataFrame:
    history = _historical_store_rows(int(request_data["Store"]), date)
    row = _request_row(request_data, date)
    combined = pd.concat([history, row], ignore_index=True)
    engineered = build_features(combined)
    values = engineered.iloc[[-1]][FEATURE_COLUMNS].copy()
    values = values.fillna(0)
    return values


def _predictions(features: pd.DataFrame) -> dict[str, float]:
    models, weights = _load_models()
    predictions = {
        name: float(models[name].predict(features)[0]) for name in MODEL_NAMES
    }
    predictions["ensemble"] = sum(weights[name] * predictions[name] for name in MODEL_NAMES)
    return {name: round(value, 2) for name, value in predictions.items()}


def _save_history(
    predictions: dict[str, float],
    store: int,
    current_user: str,
    db: Session,
    model_names: tuple[str, ...],
) -> None:
    for model_name in model_names:
        db.add(
            ForecastHistory(
                user_email=current_user,
                store=store,
                model_used=model_name,
                predicted_sales=predictions[model_name],
            )
        )
    db.commit()


@router.post("/forecast", response_model=ForecastResponse)
def forecast(
    request: ForecastRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    date = pd.Timestamp(request.Date)
    features = _features_for_request(request.model_dump(), date)
    predictions = _predictions(features)
    prediction = predictions["ensemble"]
    _save_history(predictions, request.Store, current_user, db, ("ensemble",))
    return ForecastResponse(predicted_sales=prediction)


@router.post("/models/compare", response_model=ModelComparisonResponse)
def compare_models(
    request: CompareRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    date = pd.Timestamp(request.Date)
    features = _features_for_request(request.model_dump(), date)
    predictions = _predictions(features)
    _save_history(predictions, request.Store, current_user, db, tuple(predictions))
    return ModelComparisonResponse(predictions=predictions)


@router.get("/models/metrics")
def get_model_metrics(current_user: str = Depends(get_current_user)):
    return _load_metrics()


@router.get("/history")
def get_history(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    records = db.query(ForecastHistory).filter(
        ForecastHistory.user_email == current_user
    ).order_by(ForecastHistory.requested_at.desc()).all()

    return [
        {
            "store": r.store,
            "model_used": r.model_used,
            "predicted_sales": r.predicted_sales,
            "requested_at": r.requested_at,
        }
        for r in records
    ]
