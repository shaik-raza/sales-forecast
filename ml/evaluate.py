"""Shared time-based splitting and regression metrics for model evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def time_based_split(
    data: pd.DataFrame,
    date_column: str = "Date",
    validation_weeks: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically, reserving the final weeks for validation."""
    if validation_weeks <= 0:
        raise ValueError("validation_weeks must be greater than zero")
    if date_column not in data.columns:
        raise KeyError(f"Missing date column: {date_column}")

    dates = pd.to_datetime(data[date_column], errors="raise")
    split_date = dates.max() - pd.Timedelta(weeks=validation_weeks)
    train = data.loc[dates <= split_date].copy()
    validation = data.loc[dates > split_date].copy()

    if train.empty or validation.empty:
        raise ValueError("Time-based split must produce non-empty train and validation sets")
    return train, validation


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Return MAE, RMSE, MAPE, and Rossmann RMSPE as percentages where applicable."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("actual and predicted must have the same shape")
    if y_true.ndim != 1:
        raise ValueError("actual and predicted must be one-dimensional")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("actual and predicted must contain only finite values")

    errors = y_pred - y_true
    nonzero = y_true != 0
    if not nonzero.any():
        raise ValueError("MAPE and RMSPE require at least one non-zero actual value")

    percentage_errors = errors[nonzero] / y_true[nonzero]
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE": float(np.mean(np.abs(percentage_errors)) * 100),
        "RMSPE": float(np.sqrt(np.mean(percentage_errors**2)) * 100),
    }


def evaluate_model(
    model: Any,
    features: pd.DataFrame,
    actual: pd.Series | np.ndarray,
    predict: Callable[[Any, pd.DataFrame], np.ndarray] | None = None,
) -> dict[str, float]:
    """Predict with a fitted model and score it using the shared metrics."""
    predictor = predict or (lambda fitted_model, values: fitted_model.predict(values))
    predicted = predictor(model, features)
    return regression_metrics(actual, predicted)
