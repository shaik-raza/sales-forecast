"""Tune and train the all-store XGBoost model."""

from xgboost import XGBRegressor

from ml.models.training_common import RANDOM_STATE, train_and_save


def make_model(parameters: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE,
        **parameters,
    )


def search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


if __name__ == "__main__":
    train_and_save("xgboost", make_model, search_space)
