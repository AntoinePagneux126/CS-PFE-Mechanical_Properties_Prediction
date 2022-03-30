"""This module aims to load models for inference and try it on test data."""
# pylint: disable=import-error, no-name-in-module, unused-import
import argparse
import os
import pickle
import torch
import torch.nn as nn
import yaml

from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np

import data.loader as loader
from train import generate_unique_logpath
import visualization.vis as vis
from tools.utils import load_model, compute_features_importance
from tools.valid import test_one_epoch


def inference_ml(
    cfg, average=False, model_path=None
):  # pylint: disable=too-many-locals
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration
        average (bool): Do you call this function to average different models
        model_path (str): Specify the path too the model you wwant to load.
                By default it uses the one defined in cfg
    """

    # Load data
    preprocessed_data, preprocessed_test_data, _ = loader.main(cfg=cfg)

    x_test = np.concatenate(
        (
            preprocessed_data["x_train"],
            preprocessed_data["x_valid"],
            preprocessed_test_data["x_test"],
        )
    )

    # Load model
    model_path = cfg["TEST"]["PATH_TO_MODEL"] if model_path is None else model_path
    model = pickle.load(open(model_path, "rb"))
    # Set path
    top_logdir = cfg["TEST"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, "inference")
    if not average and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Compute predictions
    y_test_true = np.concatenate(
        (
            preprocessed_data["y_train"],
            preprocessed_data["y_valid"],
            preprocessed_test_data["y_test"],
        )
    )

    y_test_pred = model.predict(x_test)

    y_true = {"test": y_test_true}
    y_pred = {"test": y_test_pred}

    if average:
        return y_true, y_pred
    # Compute metrics
    metrics = {}

    metrics["MSE_test"] = mean_squared_error(y_test_true, y_test_pred)
    metrics["RMSE_test"] = np.sqrt(metrics["MSE_test"])
    metrics["R2_test"] = r2_score(y_test_true, y_test_pred)

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


def inference_nn(
    cfg, average=False, model_path=None
):  # pylint: disable=too-many-locals
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration
        average (bool): Do you call this function to average different models
        model_path (str): Specify the path too the model you wwant to load.
                By default it uses the one defined in cfg
    """

    # Load test data
    train_loader, valid_loader, test_loader, features_name = loader.main(cfg=cfg)

    # Create on DataLoader
    dataset = ConcatDataset(
        [train_loader.dataset, valid_loader.dataset, test_loader.dataset]
    )
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )
    # Set path
    top_logdir = cfg["TEST"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, "inference")
    if not average and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Define device for computational efficiency
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Load model for inference
    input_size = test_loader.dataset[0][0].shape[0]

    # Define the loss
    f_loss = nn.MSELoss()

    # Load model
    model_path = cfg["TEST"]["PATH_TO_MODEL"] if model_path is None else model_path
    model = load_model(
        cfg=cfg,
        input_size=input_size,
        num_hidden_neuron=cfg["TRAIN"]["NUM_HIDDEN_NEURON"],
    )
    model = model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Compute Metrics
    metrics = {}

    test_loss, test_r2, y_test_true, y_test_pred = test_one_epoch(
        model, test_loader, f_loss, device, return_predictions=True
    )

    y_true = {"test": y_test_true}
    y_pred = {"test": y_test_pred}

    if average:
        return y_true, y_pred

    # Compute metrics
    metrics["MSE_test"] = test_loss
    metrics["RMSE_test"] = np.sqrt(metrics["MSE_test"])
    metrics["R2_test"] = test_r2

    print("\n###########")
    print("# Results #")
    print("###########\n")

    print(f"Test RMSE: {metrics['RMSE_test']}   | Test r2: {test_r2}")

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

    # Plot feature importance
    importance, importance_distrib = compute_features_importance(
        model=model, data=train_loader.dataset.x_data
    )

    vis.plot_feature_importance(
        importance=importance,
        names=features_name,
        model_type=cfg["TRAIN"]["MODEL"],
        path_to_save=save_dir,
        target_name=target_name,
    )

    vis.plot_feature_distrib(
        importance=importance_distrib,
        names=features_name,
        model_type=cfg["TRAIN"]["MODEL"],
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

    # Run inference
    if config_file["MODELS"]["NN"]:
        inference_nn(cfg=config_file)

    else:
        inference_ml(cfg=config_file)
