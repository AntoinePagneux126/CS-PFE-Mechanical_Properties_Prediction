"""This module aims to launch a training procedure."""
# pylint: disable=import-error, no-name-in-module, expression-not-assigned
import os
import argparse
import json
import pickle
from shutil import copyfile
import yaml
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tools.trainer import train_one_epoch
from tools.utils import load_model, choose_scheduler, launch_grid_search
from tools.valid import test_one_epoch, ModelCheckpoint
import data.loader as loader
import visualization.vis as vis


def generate_unique_logpath(logdir, raw_run_name):
    """Verify if the path already exist

    Args:
        logdir (str): path to log dir
        raw_run_name (str): name of the file

    Returns:
        str: path to the output file
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def main_ml(cfg, path_to_config):  # pylint: disable=too-many-locals
    """Main pipeline to train a ML model

    Args:
        cfg (dict): config with all the necessary parameters
        path_to_config(string): path to the config file
    """
    # Load data
    preprocessed_data, _, features_name = loader.main(cfg=cfg)

    # Init directory to save model saving best models
    top_logdir = cfg["TRAIN"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, cfg["MODELS"]["ML"]["TYPE"].lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    copyfile(path_to_config, os.path.join(save_dir, "config_file.yaml"))

    if cfg["MODELS"]["ML"]["GRID_SEARCH"]:
        model, params = launch_grid_search(cfg, preprocessed_data)

        with open(
            os.path.join(
                save_dir,
                f"best_params_{cfg['DATASET']['PREPROCESSING']['TARGET']}.json",
            ),
            "w",
        ) as outfile:
            json.dump(params, outfile, indent=2)

    else:
        model = load_model(cfg=cfg)

    model.fit(X=preprocessed_data["x_train"], y=preprocessed_data["y_train"])
    pickle.dump(model, open(os.path.join(save_dir, "model.pck"), "wb"))

    # Compute predictions
    y_pred = model.predict(preprocessed_data["x_valid"])
    y_train_true = preprocessed_data["y_train"]
    y_valid_true = preprocessed_data["y_valid"]
    y_train_pred = model.predict(preprocessed_data["x_train"])

    # Compute metrics
    metrics = {}

    metrics["MSE_train"] = mean_squared_error(y_train_true, y_train_pred)
    metrics["RMSE_train"] = np.sqrt(metrics["MSE_train"])
    metrics["R2_train"] = r2_score(y_train_true, y_train_pred)
    metrics["MSE_val"] = mean_squared_error(y_valid_true, y_pred)
    metrics["RMSE_val"] = np.sqrt(metrics["MSE_val"])
    metrics["R2_val"] = r2_score(y_valid_true, y_pred)

    # Print results
    print("\n###########")
    print("# Results #")
    print("###########\n")

    print(f"Train RMSE: {metrics['RMSE_train']} | Train r2: {metrics['R2_train']}")
    print(f"Valid RMSE: {metrics['RMSE_val']} | Valid r2: {metrics['R2_val']}")

    y_true = {"train": y_train_true, "valid": y_valid_true}
    y_pred = {"train": y_train_pred, "valid": y_pred}

    # Plot and save resuslts
    target_name = "$R_{m}$"
    if cfg["DATASET"]["PREPROCESSING"]["TARGET"] == "re02":
        target_name = "$R_{e02}$"
    elif cfg["DATASET"]["PREPROCESSING"]["TARGET"] == "A80":
        target_name = "$A_{80}$"

    vis.plot_partial_y_pred_y_true(
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
        path_to_save=save_dir,
        target_name=target_name,
    )

    # (Plot and) Save the features importance
    if cfg["MODELS"]["ML"]["TYPE"] in [
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
    ]:
        importances = model.feature_importances_
        vis.plot_feature_importance(
            importances,
            features_name,
            cfg["MODELS"]["ML"]["TYPE"],
            save_dir,
            target_name,
        )


def main_nn(
    cfg, path_to_config
):  # pylint: disable=too-many-locals, too-many-statements
    """Main pipeline to train a model

    Args:
        cfg (dict): config with all the necessary parameters
        path_to_config(string): path to the config file
    """

    # Load data
    train_loader, valid_loader, test_loader = loader.main(cfg=cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define the model
    input_size = train_loader.dataset[0][0].shape[0]

    model = load_model(
        cfg=cfg,
        input_size=input_size,
        num_hidden_neuron=cfg["TRAIN"]["NUM_HIDDEN_NEURON"],
    )
    model = model.to(device)

    # Define the loss
    f_loss = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["TRAIN"]["LR"]["LR_INITIAL"]
    )

    # Tracking with tensorboard
    tensorboard_writer = SummaryWriter(log_dir=cfg["TRAIN"]["LOG_DIR"])

    # Init directory to save model saving best models
    top_logdir = cfg["TRAIN"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, cfg["TRAIN"]["MODEL"].lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    copyfile(path_to_config, os.path.join(save_dir, "config_file.yaml"))

    # Init Checkpoint class
    checkpoint = ModelCheckpoint(
        save_dir, model, cfg["TRAIN"]["EPOCH"], cfg["TRAIN"]["CHECKPOINT_STEP"]
    )

    # Lr scheduler
    scheduler = choose_scheduler(cfg=cfg["TRAIN"]["LR"], optimizer=optimizer)

    # Launch training loop
    for epoch in range(cfg["TRAIN"]["EPOCH"]):
        print("EPOCH : {}".format(epoch))

        train_loss, train_r2 = train_one_epoch(
            model, train_loader, f_loss, optimizer, device
        )
        val_loss, val_r2 = test_one_epoch(model, valid_loader, f_loss, device)

        # Update learning rate
        scheduler.step() if cfg["TRAIN"]["LR"][
            "TYPE"
        ] == "CyclicLR" else scheduler.step(val_r2)
        learning_rate = scheduler.optimizer.param_groups[0]["lr"]

        # Save best checkpoint
        checkpoint.update(val_loss, epoch)

        # Track performances with tensorboard
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_loss"), train_loss, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "train_r2"), train_r2, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_loss"), val_loss, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "val_r2"), val_r2, epoch
        )
        tensorboard_writer.add_scalar(
            os.path.join(cfg["TRAIN"]["LOG_DIR"], "lr"), learning_rate, epoch
        )

    # Compute Metrics
    metrics = {}

    train_loss, train_r2, y_train_true, y_train_pred = test_one_epoch(
        model, train_loader, f_loss, device, return_predictions=True
    )
    val_loss, val_r2, y_valid_true, y_valid_pred = test_one_epoch(
        model, valid_loader, f_loss, device, return_predictions=True
    )
    test_loss, test_r2, y_test_true, y_test_pred = test_one_epoch(
        model, test_loader, f_loss, device, return_predictions=True
    )

    metrics["MSE_train"] = train_loss
    metrics["RMSE_train"] = np.sqrt(metrics["MSE_train"])
    metrics["R2_train"] = train_r2

    metrics["MSE_val"] = val_loss
    metrics["RMSE_val"] = np.sqrt(metrics["MSE_val"])
    metrics["R2_val"] = val_r2

    metrics["MSE_test"] = test_loss
    metrics["RMSE_test"] = np.sqrt(metrics["MSE_test"])
    metrics["R2_test"] = test_r2

    print("\n###########")
    print("# Results #")
    print("###########\n")

    print(f"Train RMSE: {metrics['RMSE_train']} | Train r2: {train_r2}")
    print(f"Valid RMSE: {metrics['RMSE_val']}   | Valid r2: {val_r2}")
    print(f"Test RMSE: {metrics['RMSE_test']}   | Test r2: {test_r2}")

    y_true = {"train": y_train_true, "valid": y_valid_true, "test": y_test_true}
    y_pred = {"train": y_train_pred, "valid": y_valid_pred, "test": y_test_pred}

    # Plot and save resuslts
    target_name = "$R_{m}$"
    if cfg["DATASET"]["PREPROCESSING"]["TARGET"] == "re02":
        target_name = "$R_{e02}$"
    elif cfg["DATASET"]["PREPROCESSING"]["TARGET"] == "A80":
        target_name = "$A_{80}$"
    vis.plot_all_y_pred_y_true(
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
        path_to_save=save_dir,
        target_name=target_name,
    )


if __name__ == "__main__":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the config file to the command line arguments
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file",
    )
    args = parser.parse_args()

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    if config_file["MODELS"]["NN"]:
        main_nn(cfg=config_file, path_to_config=args.path_to_config)

    else:
        main_ml(cfg=config_file, path_to_config=args.path_to_config)
