# pylint: disable=invalid-name
"""Create Machine Learning Regressor models."""
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)


def models(cfg):
    """Return a ML Regression model

    Args:
        cfg (dict): configuration file

    Returns:
        sklearn.Model: ML model
    """

    target_to_predict = cfg["DATASET"]["PREPROCESSING"]["TARGET"]

    if cfg["MODELS"]["ML"]["TYPE"] == "RandomForest":
        model = RandomForestRegressor(
            **cfg["MODELS"]["ML"]["RandomForest"][target_to_predict]
        )
    if cfg["MODELS"]["ML"]["TYPE"] == "ExtraTrees":
        model = ExtraTreesRegressor(
            **cfg["MODELS"]["ML"]["ExtraTrees"][target_to_predict]
        )
    if cfg["MODELS"]["ML"]["TYPE"] == "GradientBoosting":
        model = GradientBoostingRegressor(
            **cfg["MODELS"]["ML"]["GradientBoosting"][target_to_predict]
        )

    return model
