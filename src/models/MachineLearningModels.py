# pylint: disable=invalid-name
"""Create Machine Learning Regressor models."""
from sklearn.ensemble import RandomForestRegressor


def models(cfg):
    """Return a ML Regression model

    Args:
        cfg (dict): configuration file

    Returns:
        sklearn.Model: ML model
    """
    if cfg["MODELS"]["ML"]["TYPE"] == "RandomForest":
        model = RandomForestRegressor(**cfg["MODELS"]["ML"]["RandomForest"])

    return model
