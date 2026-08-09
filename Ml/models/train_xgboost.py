import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Load cleaned data
df = pd.read_csv("Data/processed/train_merged_clean.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date - important for time series
df = df.sort_values('Date')

# Features to use for prediction
features = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance',
    'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
    'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
    'Year', 'Month', 'Day', 'WeekOfYear'
]
target = 'Sales'

X = df[features]
y = df[target]

# Time-based split: last 6 weeks as test set (not random shuffling!)
split_date = df['Date'].max() - pd.Timedelta(weeks=6)
train_mask = df['Date'] <= split_date
test_mask = df['Date'] > split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Train model
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Save model
os.makedirs("ml/artifacts", exist_ok=True)
joblib.dump(model, "ml/artifacts/xgboost_model.pkl")
print("Model saved to ml/artifacts/xgboost_model.pkl")