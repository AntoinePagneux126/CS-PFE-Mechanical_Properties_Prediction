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
    preprocessed_data, _ = loader.main(cfg=cfg)

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

    y_pred = model.predict(preprocessed_data["x_valid"])

    train_loss = mean_squared_error(
        preprocessed_data["y_train"], model.predict(preprocessed_data["x_train"])
    )
    train_r2 = r2_score(
        preprocessed_data["y_train"], model.predict(preprocessed_data["x_train"])
    )
    val_loss = mean_squared_error(preprocessed_data["y_valid"], y_pred)
    val_r2 = r2_score(preprocessed_data["y_valid"], y_pred)

    print("\n###########")
    print("# Results #")
    print("###########\n")

    print(f"Train RMSE: {np.sqrt(train_loss)} | Train r2: {train_r2}")
    print(f"Valid RMSE: {np.sqrt(val_loss)} | Valid r2: {val_r2}")


def main_nn(cfg, path_to_config):  # pylint: disable=too-many-locals
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
    train_loss, train_r2, y_train_true, y_train_pred = test_one_epoch(
        model, train_loader, f_loss, device, return_predictions=True
    )
    val_loss, val_r2, y_valid_true, y_valid_pred = test_one_epoch(
        model, valid_loader, f_loss, device, return_predictions=True
    )
    test_loss, test_r2, y_test_true, y_test_pred = test_one_epoch(
        model, test_loader, f_loss, device, return_predictions=True
    )

    print("\n###########")
    print("# Results #")
    print("###########\n")

    print(f"Train MSE: {train_loss} | Train r2: {train_r2}")
    print(f"Valid MSE: {val_loss} | Valid r2: {val_r2}")
    print(f"Test MSE: {test_loss} | Test r2: {test_r2}")

    y_true = {"train": y_train_true, "valid": y_valid_true, "test": y_test_true}
    y_pred = {"train": y_train_pred, "valid": y_valid_pred, "test": y_test_pred}

    vis.plot_all_y_pred_y_true(y_true=y_true, y_pred=y_pred, path_to_save=save_dir)


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
