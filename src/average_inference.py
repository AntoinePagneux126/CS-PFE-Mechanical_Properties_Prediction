"""This module aims to load models for inference and try it on test data."""
# pylint: disable=import-error, no-name-in-module, unused-import
import argparse
import pickle
import sys
import yaml

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

import data.loader as loader
from inference import inference_nn, inference_ml


def model_average(cfg):
    """Compute the average prediction

    Args:
        cfg (dict): configuration
    """

    if not cfg["TEST"]["AVERAGE"]["ACTIVE"]:
        print("You should use inference.py !")
        sys.exit()

    # Compute probabilities for every models
    models_predictions = {"train": [], "valid": [], "test": []}
    for _, elem in enumerate(cfg["TEST"]["AVERAGE"]["PATH"]):
        with open(elem["CONFIG"], "r") as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.Loader)

        if config_file["MODELS"]["NN"]:
            y_true, y_pred = inference_nn(
                cfg=config_file, average=True, model_path=elem["MODEL"]
            )
        else:
            y_true, y_pred = inference_ml(
                cfg=config_file, average=True, model_path=elem["MODEL"]
            )

        models_predictions["train"].append(y_pred["train"].reshape(-1, 1))
        models_predictions["valid"].append(y_pred["valid"].reshape(-1, 1))
        models_predictions["test"].append(y_pred["test"].reshape(-1, 1))

    # Compute mean prediction
    y_train_pred = np.mean(np.concatenate(models_predictions["train"], axis=1), axis=1)
    y_valid_pred = np.mean(np.concatenate(models_predictions["valid"], axis=1), axis=1)
    y_test_pred = np.mean(np.concatenate(models_predictions["test"], axis=1), axis=1)
    y_pred = {"train": y_train_pred, "valid": y_valid_pred, "test": y_test_pred}

    # Compute metrics
    metrics = {}
    metrics["MSE_train"] = mean_squared_error(y_true["train"], y_train_pred)
    metrics["RMSE_train"] = np.sqrt(metrics["MSE_train"])
    metrics["R2_train"] = r2_score(y_true["train"], y_train_pred)
    metrics["MSE_val"] = mean_squared_error(y_true["valid"], y_valid_pred)
    metrics["RMSE_val"] = np.sqrt(metrics["MSE_val"])
    metrics["R2_val"] = r2_score(y_true["valid"], y_valid_pred)
    metrics["MSE_test"] = mean_squared_error(y_true["test"], y_test_pred)
    metrics["RMSE_test"] = np.sqrt(metrics["MSE_test"])
    metrics["R2_test"] = r2_score(y_true["test"], y_test_pred)

    # Print results
    print("\n###########")
    print("# Results #")
    print("###########\n")

    print(f"Train RMSE: {metrics['RMSE_train']} | Train r2: {metrics['R2_train']}")
    print(f"Valid RMSE: {metrics['RMSE_val']} | Valid r2: {metrics['R2_val']}")
    print(f"Test RMSE: {metrics['RMSE_test']} | Valid r2: {metrics['R2_test']}")


if __name__ == "__main__":
    # Init the parser;
    inference_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add path to the config file to the command line arguments;
    inference_parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file.",
    )
    args = inference_parser.parse_args()

    # Load config file
    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    # Run model average
    model_average(cfg=config_file)
