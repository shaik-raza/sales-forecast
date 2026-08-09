from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import joblib
import pandas as pd
from api.forecast.schemas import ForecastRequest, ForecastResponse, CompareRequest, ModelComparisonResponse
from api.auth.jwt_handler import get_current_user
from api.db.database import get_db, engine, Base
from api.db.models import ForecastHistory

Base.metadata.create_all(bind=engine)

router = APIRouter()

xgb_model = joblib.load("ml/artifacts/xgboost_model.pkl")
store_df = pd.read_csv("Data/raw/store.csv")

storetype_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
assortment_map = {'a': 1, 'b': 2, 'c': 3}


@router.post("/forecast", response_model=ForecastResponse)
def forecast(
    request: ForecastRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    input_data = pd.DataFrame([request.model_dump()])
    prediction = round(float(xgb_model.predict(input_data)[0]), 2)

    history_entry = ForecastHistory(
        user_email=current_user,
        store=request.Store,
        model_used="xgboost",
        predicted_sales=prediction
    )
    db.add(history_entry)
    db.commit()

    return ForecastResponse(predicted_sales=prediction)


@router.post("/models/compare", response_model=ModelComparisonResponse)
def compare_models(
    request: CompareRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    predictions = {}

    if request.Store != 1:
        raise HTTPException(
            status_code=400,
            detail="Comparison currently only supported for Store 1 (Prophet/SARIMA trained on this store only)"
        )

    date = pd.to_datetime(request.Date)
    store_row = store_df[store_df['Store'] == request.Store].iloc[0]

    xgb_input = pd.DataFrame([{
        'Store': request.Store,
        'DayOfWeek': date.dayofweek + 1,
        'Promo': 0,
        'StateHoliday': 0,
        'SchoolHoliday': 0,
        'StoreType': storetype_map.get(store_row['StoreType'], 0),
        'Assortment': assortment_map.get(store_row['Assortment'], 0),
        'CompetitionDistance': store_row['CompetitionDistance'] if pd.notna(store_row['CompetitionDistance']) else 0,
        'CompetitionOpenSinceMonth': store_row['CompetitionOpenSinceMonth'] if pd.notna(store_row['CompetitionOpenSinceMonth']) else 0,
        'CompetitionOpenSinceYear': store_row['CompetitionOpenSinceYear'] if pd.notna(store_row['CompetitionOpenSinceYear']) else 0,
        'Promo2': store_row['Promo2'],
        'Promo2SinceWeek': store_row['Promo2SinceWeek'] if pd.notna(store_row['Promo2SinceWeek']) else 0,
        'Promo2SinceYear': store_row['Promo2SinceYear'] if pd.notna(store_row['Promo2SinceYear']) else 0,
        'Year': date.year,
        'Month': date.month,
        'Day': date.day,
        'WeekOfYear': date.isocalendar()[1],
    }])
    predictions["xgboost"] = round(float(xgb_model.predict(xgb_input)[0]), 2)

    prophet_model = joblib.load("ml/artifacts/prophet_store_1.pkl")
    future = pd.DataFrame({"ds": [date]})
    prophet_forecast = prophet_model.predict(future)
    predictions["prophet"] = round(float(prophet_forecast["yhat"].iloc[0]), 2)

    sarima_model = joblib.load("ml/artifacts/sarima_store_1.pkl")
    sarima_forecast = sarima_model.forecast(steps=1)
    predictions["sarima"] = round(float(sarima_forecast.iloc[-1]), 2)

    # Save each model's prediction as a separate history entry
    for model_name, pred_value in predictions.items():
        history_entry = ForecastHistory(
            user_email=current_user,
            store=request.Store,
            model_used=model_name,
            predicted_sales=pred_value
        )
        db.add(history_entry)
    db.commit()

    return ModelComparisonResponse(predictions=predictions)


@router.get("/history")
def get_history(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    records = db.query(ForecastHistory).filter(
        ForecastHistory.user_email == current_user
    ).order_by(ForecastHistory.requested_at.desc()).all()

    return [
        {
            "store": r.store,
            "model_used": r.model_used,
            "predicted_sales": r.predicted_sales,
            "requested_at": r.requested_at
        }
        for r in records
    ]