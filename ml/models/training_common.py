"""Shared data preparation and tuning helpers for model training scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ml.evaluate import regression_metrics, time_based_split
from ml.features import FEATURE_COLUMNS, build_features

DATA_PATH = Path("Data/processed/train_merged_clean.csv")
ARTIFACT_DIR = Path("ml/artifacts")
TARGET = "Sales"
RANDOM_STATE = 42
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "30"))


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build the same chronological train/validation data for every model."""
    data = pd.read_csv(DATA_PATH)
    engineered = build_features(data)
    train_engineered, validation_engineered = time_based_split(engineered)
    train_features = train_engineered[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    validation_features = validation_engineered[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    train_valid = train_features.notna().all(axis=1)
    validation_valid = validation_features.notna().all(axis=1)
    return (
        train_features.loc[train_valid].reset_index(drop=True),
        validation_features.loc[validation_valid].reset_index(drop=True),
        train_engineered.loc[train_valid, TARGET].reset_index(drop=True),
        validation_engineered.loc[validation_valid, TARGET].reset_index(drop=True),
    )


def tune_model(
    make_model: Callable[[dict[str, Any]], Any],
    search_space: Callable[[optuna.Trial], dict[str, Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    """Tune a model on chronological folds and return its best parameters."""
    sample_size = min(len(x_train), 120_000)
    sample_start = len(x_train) - sample_size
    x_tune = x_train.iloc[sample_start:]
    y_tune = y_train.iloc[sample_start:]
    splitter = TimeSeriesSplit(n_splits=3)

    def objective(trial: optuna.Trial) -> float:
        parameters = search_space(trial)
        fold_scores: list[float] = []
        for fold_train, fold_validation in splitter.split(x_tune):
            model = make_model(parameters)
            model.fit(x_tune.iloc[fold_train], y_tune.iloc[fold_train])
            predictions = model.predict(x_tune.iloc[fold_validation])
            fold_scores.append(regression_metrics(y_tune.iloc[fold_validation], predictions)["RMSPE"])
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def train_and_save(
    name: str,
    make_model: Callable[[dict[str, Any]], Any],
    search_space: Callable[[optuna.Trial], dict[str, Any]],
) -> None:
    """Tune, fit, evaluate, and persist one model."""
    x_train, x_validation, y_train, y_validation = load_split()
    best_parameters = tune_model(make_model, search_space, x_train, y_train)
    model = make_model(best_parameters)
    model.fit(x_train, y_train)
    metrics = regression_metrics(y_validation, model.predict(x_validation))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "metrics": metrics,
            "parameters": best_parameters,
        },
        ARTIFACT_DIR / f"{name}.joblib",
        compress=3,
    )
    print(f"{name}: {metrics}")
    print(f"Saved to {ARTIFACT_DIR / f'{name}.joblib'}")
