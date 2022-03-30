"""This module aims to load models for inference and try it on test data."""
# pylint: disable=import-error, no-name-in-module, unused-import
import argparse
import os
import sys
import yaml

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

import visualization.vis as vis
from inference import inference_nn, inference_ml
from train import generate_unique_logpath


def model_average(cfg):
    """Compute the average prediction

    Args:
        cfg (dict): configuration
    """

    if not cfg["TEST"]["AVERAGE"]["ACTIVE"]:
        print("You should use inference.py !")
        sys.exit()

    top_logdir = cfg["TEST"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, "inference")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Compute probabilities for every models
    models_predictions = {"test": []}
    for _, elem in enumerate(cfg["TEST"]["AVERAGE"]["PATH"]):
        with open(elem["CONFIG"], "r") as ymlfile:
            config_file = yaml.load(ymlfile, Loader=yaml.Loader)
        config_file["DATASET"]["PREPROCESSING"]["MERGE_FILES"]["WHICH"] = cfg[
            "DATASET"
        ]["PREPROCESSING"]["MERGE_FILES"]["WHICH"]
        if config_file["MODELS"]["NN"]:
            y_true, y_pred = inference_nn(
                cfg=config_file, average=True, model_path=elem["MODEL"]
            )
        else:
            y_true, y_pred = inference_ml(
                cfg=config_file, average=True, model_path=elem["MODEL"]
            )

        models_predictions["test"].append(y_pred["test"].reshape(-1, 1))

    # Compute mean prediction
    y_test_pred = np.mean(np.concatenate(models_predictions["test"], axis=1), axis=1)
    y_pred = {"test": y_test_pred}

    # Compute metrics
    metrics = {}
    metrics["MSE_test"] = mean_squared_error(y_true["test"], y_test_pred)
    metrics["RMSE_test"] = np.sqrt(metrics["MSE_test"])
    metrics["R2_test"] = r2_score(y_true["test"], y_test_pred)

    # Print results
    print("\n###########")
    print("# Results #")
    print("###########\n")

    print(f"Test RMSE: {metrics['RMSE_test']} | Valid r2: {metrics['R2_test']}")

    # Plot and save resuslts
    target_name = "$R_{m}$"
    if cfg["DATASET"]["PREPROCESSING"]["TARGET"] == "re02":
        target_name = "$R_{e02}$"
    elif cfg["DATASET"]["PREPROCESSING"]["TARGET"] == "A80":
        target_name = "$A_{80}$"

    vis.plot_y_pred_y_true(
        y_true=y_true["test"],
        y_pred=y_pred["test"],
        metrics=metrics,
        path_to_save=save_dir,
        target_name=target_name,
    )
    return y_true, y_pred, metrics


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
