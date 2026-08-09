import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Load cleaned data
df = pd.read_csv("Data/processed/train_merged_clean.csv")
df['Date'] = pd.to_datetime(df['Date'])

# SARIMA also works on ONE store's time series at a time, like Prophet
STORE_ID = 1
store_df = df[df['Store'] == STORE_ID].sort_values('Date')
store_df = store_df.set_index('Date')['Sales']

# Time-based split: last 6 weeks as test set
split_date = store_df.index.max() - pd.Timedelta(weeks=6)
train_series = store_df[store_df.index <= split_date]
test_series = store_df[store_df.index > split_date]

print(f"Train size: {len(train_series)}, Test size: {len(test_series)}")

# SARIMA(p,d,q)(P,D,Q,s) - using weekly seasonality (s=7, since sales patterns repeat weekly)
model = SARIMAX(
    train_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted_model = model.fit(disp=False)

# Forecast
forecast = fitted_model.forecast(steps=len(test_series))

# Evaluate
mae = mean_absolute_error(test_series, forecast)
rmse = np.sqrt(mean_squared_error(test_series, forecast))
mape = np.mean(np.abs((test_series.values - forecast.values) / test_series.values)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Save model
os.makedirs("ml/artifacts", exist_ok=True)
joblib.dump(fitted_model, f"ml/artifacts/sarima_store_{STORE_ID}.pkl")
print(f"Model saved to ml/artifacts/sarima_store_{STORE_ID}.pkl")