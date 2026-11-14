"""Save and display the common validation metrics for every model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from ml.models.train_ensemble import MODEL_NAMES
from ml.models.training_common import ARTIFACT_DIR


def main() -> None:
    results: dict[str, dict[str, float | bool]] = {}
    for name in MODEL_NAMES:
        artifact = joblib.load(ARTIFACT_DIR / f"{name}.joblib")
        results[name] = artifact["metrics"]

    ensemble = joblib.load(ARTIFACT_DIR / "ensemble.joblib")
    results["ensemble"] = {
        **ensemble["metrics"],
        "beats_every_single_model": ensemble["beats_every_single_model"],
    }

    ordered = dict(sorted(results.items(), key=lambda item: item[1]["RMSPE"]))
    output_path = ARTIFACT_DIR / "model_metrics.json"
    output_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")

    print(f"{'Model':<16} {'MAE':>12} {'RMSE':>12} {'MAPE':>10} {'RMSPE':>10}")
    print("-" * 64)
    for name, metrics in ordered.items():
        print(
            f"{name:<16} {metrics['MAE']:>12.2f} {metrics['RMSE']:>12.2f} "
            f"{metrics['MAPE']:>9.2f}% {metrics['RMSPE']:>9.2f}%"
        )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
