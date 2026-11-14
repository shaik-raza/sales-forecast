"""Tune and train the all-store LightGBM model."""

from lightgbm import LGBMRegressor

from ml.models.training_common import RANDOM_STATE, train_and_save


def make_model(parameters: dict) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_jobs=2,
        verbosity=-1,
        random_state=RANDOM_STATE,
        **parameters,
    )


def search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
    }


if __name__ == "__main__":
    train_and_save("lightgbm", make_model, search_space)
