"""Tune and train the all-store CatBoost model."""

from catboost import CatBoostRegressor

from ml.models.training_common import RANDOM_STATE, train_and_save


def make_model(parameters: dict) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        thread_count=2,
        verbose=False,
        random_seed=RANDOM_STATE,
        **parameters,
    )


def search_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 150, 400, step=50),
        "depth": trial.suggest_int("depth", 5, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


if __name__ == "__main__":
    train_and_save("catboost", make_model, search_space)
