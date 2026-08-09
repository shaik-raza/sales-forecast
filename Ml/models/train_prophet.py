import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os

# Load cleaned data
df = pd.read_csv("Data/processed/train_merged_clean.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Prophet works on ONE time series at a time - let's pick a single store to start
# (later we can loop this for all stores, but one store proves the concept first)
STORE_ID = 1
store_df = df[df['Store'] == STORE_ID].sort_values('Date')

# Prophet requires exactly two columns: 'ds' (date) and 'y' (value to predict)
prophet_df = store_df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})

# Time-based split: last 6 weeks as test set
split_date = prophet_df['ds'].max() - pd.Timedelta(weeks=6)
train_df = prophet_df[prophet_df['ds'] <= split_date]
test_df = prophet_df[prophet_df['ds'] > split_date]

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Train model
model = Prophet()
model.fit(train_df)

# Predict on test period
future = test_df[['ds']]
forecast = model.predict(future)

# Evaluate
y_true = test_df['y'].values
y_pred = forecast['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Save model
os.makedirs("ml/artifacts", exist_ok=True)
joblib.dump(model, f"ml/artifacts/prophet_store_{STORE_ID}.pkl")
print(f"Model saved to ml/artifacts/prophet_store_{STORE_ID}.pkl")