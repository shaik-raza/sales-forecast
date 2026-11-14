# Sales Forecast API

FastAPI sales forecasting for the Rossmann Store Sales dataset.

## Models

The project trains four all-store regressors with the same chronological split
and leakage-safe features:

- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Inverse-RMSPE weighted ensemble

Models are saved under `ml/artifacts/`. The API loads them lazily on the first
forecast request.

## Evaluation metrics

The shared evaluator reports:

- MAE
- RMSE
- MAPE
- RMSPE, Rossmann's metric, ignoring zero-sales rows
- R²

Regenerate the comparison table and JSON with:

```powershell
.\venv\Scripts\python.exe -m ml.save_metrics
```

The results are saved to `ml/artifacts/model_metrics.json` and exposed through
the authenticated `GET /models/metrics` endpoint.

## API

Run locally:

```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Available endpoints include:

- `POST /signup`
- `POST /login`
- `POST /forecast`
- `POST /models/compare`
- `GET /models/metrics`
- `GET /history`
- `GET /health`

`/forecast` and `/models/compare` use the same request schema and shared
feature-building path, including real historical store sales for lag and
rolling features.

## Run the frontend

```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/` and create an account or sign in.

| Model | MAE | MAPE | RMSPE | R² |
|---|---:|---:|---:|---:|
| Ensemble | €606.35 | 9.26% | 13.37% | 0.9183 |
| XGBoost | €611.00 | 9.32% | 13.37% | 0.9175 |
| LightGBM | €613.38 | 9.38% | 13.44% | 0.9171 |
| CatBoost | €606.47 | 9.30% | 13.51% | 0.9183 |
| Random Forest | €659.86 | 10.06% | 14.66% | 0.9020 |
