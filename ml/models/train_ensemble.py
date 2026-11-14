"""Build and evaluate an inverse-RMSPE weighted ensemble."""

from __future__ import annotations

import joblib
import numpy as np

from ml.evaluate import regression_metrics
from ml.models.training_common import ARTIFACT_DIR, load_split


MODEL_NAMES = ("xgboost", "lightgbm", "catboost", "random_forest")


def main() -> None:
    x_train, x_validation, y_train, y_validation = load_split()
    del x_train, y_train

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES:
        artifact = joblib.load(ARTIFACT_DIR / f"{name}.joblib")
        prediction = artifact["model"].predict(x_validation)
        predictions[name] = prediction
        metrics[name] = artifact["metrics"]

    inverse_scores = {name: 1.0 / values["RMSPE"] for name, values in metrics.items()}
    denominator = sum(inverse_scores.values())
    weights = {name: score / denominator for name, score in inverse_scores.items()}
    ensemble_prediction = sum(weights[name] * predictions[name] for name in MODEL_NAMES)
    ensemble_metrics = regression_metrics(y_validation, ensemble_prediction)
    beats_all = all(
        ensemble_metrics["RMSPE"] < values["RMSPE"] for values in metrics.values()
    )

    artifact = {
        "model_names": list(MODEL_NAMES),
        "weights": weights,
        "metrics": ensemble_metrics,
        "beats_every_single_model": beats_all,
    }
    joblib.dump(artifact, ARTIFACT_DIR / "ensemble.joblib", compress=3)

    print("Ensemble weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    print(f"ensemble: {ensemble_metrics}")
    print(f"Beats every single model: {beats_all}")
    print(f"Saved to {ARTIFACT_DIR / 'ensemble.joblib'}")


if __name__ == "__main__":
    main()
