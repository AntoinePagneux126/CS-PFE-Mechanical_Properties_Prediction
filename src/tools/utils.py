"""This module aims to define utils function for the project."""
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from torch.optim import lr_scheduler

from models.LinearNet_1 import LinearNet_1
from models.LinearNet_2 import LinearNet_2
from models.LinearNet_3 import LinearNet_3
from models.MachineLearningModels import models


def load_model(cfg, input_size=100, num_hidden_neuron=64):
    """This function aims to load the right model regarding the configuration file

    Args:
        cfg (dict): Configuration file

    Returns:
        nn.Module: Neural Network
    """
    if cfg["MODELS"]["NN"]:
        if cfg["TRAIN"]["MODEL"] == "LinearNet_1":
            return LinearNet_1(
                num_features=input_size, num_hidden_neuron=num_hidden_neuron
            )
        elif cfg["TRAIN"]["MODEL"] == "LinearNet_2":
            return LinearNet_2(num_features=input_size)
        elif cfg["TRAIN"]["MODEL"] == "LinearNet_3":
            return LinearNet_3(
                num_features=input_size, num_hidden_neuron=num_hidden_neuron
            )
        else:
            return LinearNet_1(
                num_features=input_size, num_hidden_neuron=num_hidden_neuron
            )
    else:
        return models(cfg=cfg)


def choose_scheduler(cfg, optimizer):
    """This function aims to choose between different learning rate scheduler

    Args:
        cfg (dict): Configuration file

    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler
    """
    if cfg["TYPE"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg["ReduceLROnPlateau"]["LR_DECAY"],
            patience=cfg["ReduceLROnPlateau"]["LR_PATIENCE"],
            threshold=cfg["ReduceLROnPlateau"]["LR_THRESHOLD"],
        )
    elif cfg["TYPE"] == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg["CyclicLR"]["BASE_LR"],
            max_lr=cfg["CyclicLR"]["MAX_LR"],
            step_size_up=cfg["CyclicLR"]["STEP_SIZE"],
            cycle_momentum=False,
        )
    return scheduler


def launch_grid_search(cfg, preprocessed_data):  # pylint: disable=too-many-locals
    """Launch a grid search on different models

    Args:
        cfg (dict): Configuration file
        preprocessed_data (dict): data
    """
    # Train
    x_train = preprocessed_data["x_train"]
    y_train = preprocessed_data["y_train"]
    # Valid
    x_valid = preprocessed_data["x_valid"]
    y_valid = preprocessed_data["y_valid"]

    if cfg["MODELS"]["ML"]["TYPE"] == "RandomForest":
        rfr = RandomForestRegressor(bootstrap=False, n_jobs=-1)

        param_grid = {
            "min_samples_split": np.arange(4, 9, 2),
            "max_depth": np.arange(20, 28, 1),
            "max_features": np.arange(12, min(x_train.shape[1], 18), 1),
            "n_estimators": np.arange(70, 120, 5),
        }

        rfr_cv = GridSearchCV(rfr, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)
        rfr_cv.fit(
            np.concatenate((x_train, x_valid)), np.concatenate((y_train, y_valid))
        )

        params = {}
        for key, value in rfr_cv.best_params_.items():
            params[key] = int(value)

        return rfr_cv.best_estimator_, params

    else:
        model = RandomForestRegressor(
            bootstrap=False,
            max_depth=23,
            max_features=12,
            min_samples_split=6,
            n_estimators=75,
            n_jobs=-1,
        )
        params = {
            "max_depth": 23,
            "max_features": 12,
            "min_samples_split": 6,
            "n_estimators": 75,
            "bootstrap": False,
            "n_jobs": 1,
        }
        return model, params
