"""Tune and train the all-store Random Forest model."""

from sklearn.ensemble import RandomForestRegressor

from ml.models.training_common import RANDOM_STATE, train_and_save


def make_model(parameters: dict) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_jobs=2,
        random_state=RANDOM_STATE,
        max_samples=0.1,
        **parameters,
    )


def search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 25, 50, step=25),
        "max_depth": trial.suggest_int("max_depth", 8, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_float("max_features", 0.6, 1.0),
    }


if __name__ == "__main__":
    train_and_save("random_forest", make_model, search_space)
