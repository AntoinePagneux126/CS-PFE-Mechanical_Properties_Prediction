"""This module aims to load and process the data."""
# pylint: disable=import-error, no-name-in-module
import argparse
import torch
import yaml

import numpy as np

from torch.utils.data import DataLoader
from data.dataset_utils import basic_random_split, RegressionDataset


def main(cfg):  # pylint: disable=too-many-locals
    """Main function to call to load and process data

    Args:
        cfg (dict): configuration file

    Returns:
        tuple[DataLoader, DataLoader]: train and validation DataLoader
        DataLoader: test DataLoader
    """

    # Set data path
    path_to_data = cfg["DATA_DIR"]

    # Load the dataset for the training/validation/test sets
    data = basic_random_split(
        path_to_data=path_to_data,
        test_valid_ratio=cfg["DATASET"]["TEST_VALID_RATIO"],
        preprocessing=cfg["DATASET"]["PREPROCESSING"]["NORMALIZE"]["TYPE"],
    )

    target_to_predict = cfg["DATASET"]["PREPROCESSING"]["TARGET"]

    if cfg["DATASET"]["PREPROCESSING"]["MERGE_FILES"]["ACTIVE"]:
        x_train, y_train = None, None
        x_valid, y_valid = None, None
        x_test, y_test = None, None
        for key in data:
            # Train
            x_train = (
                np.concatenate(
                    (x_train, data[key]["dataset"][target_to_predict]["x_train"]),
                    axis=0,
                )
                if x_train is not None
                else data[key]["dataset"][target_to_predict]["x_train"]
            )
            y_train = (
                np.concatenate(
                    (y_train, data[key]["dataset"][target_to_predict]["y_train"]),
                    axis=0,
                )
                if y_train is not None
                else data[key]["dataset"][target_to_predict]["y_train"]
            )
            # Valid
            x_valid = (
                np.concatenate(
                    (x_valid, data[key]["dataset"][target_to_predict]["x_valid"]),
                    axis=0,
                )
                if x_valid is not None
                else data[key]["dataset"][target_to_predict]["x_valid"]
            )
            y_valid = (
                np.concatenate(
                    (y_valid, data[key]["dataset"][target_to_predict]["y_valid"]),
                    axis=0,
                )
                if y_valid is not None
                else data[key]["dataset"][target_to_predict]["y_valid"]
            )
            # Test
            x_test = (
                np.concatenate(
                    (x_test, data[key]["dataset"][target_to_predict]["x_test"]), axis=0
                )
                if x_test is not None
                else data[key]["dataset"][target_to_predict]["x_test"]
            )
            y_test = (
                np.concatenate(
                    (y_test, data[key]["dataset"][target_to_predict]["y_test"]), axis=0
                )
                if y_test is not None
                else data[key]["dataset"][target_to_predict]["y_test"]
            )

    else:
        dataset = data[cfg["DATASET"]["PREPROCESSING"]["MERGE_FILES"]["WHICH"]]
        # Train
        x_train = dataset["dataset"][target_to_predict]["x_train"]
        y_train = dataset["dataset"][target_to_predict]["y_train"]
        # Valid
        x_valid = dataset["dataset"][target_to_predict]["x_valid"]
        y_valid = dataset["dataset"][target_to_predict]["y_valid"]
        # Test
        x_test = dataset["dataset"][target_to_predict]["x_test"]
        y_test = dataset["dataset"][target_to_predict]["y_test"]

    # Create train, valid and test dataset
    train_dataset = RegressionDataset(
        x_data=torch.from_numpy(x_train).float(),
        y_data=torch.from_numpy(y_train).float(),
    )
    valid_dataset = RegressionDataset(
        x_data=torch.from_numpy(x_valid).float(),
        y_data=torch.from_numpy(y_valid).float(),
    )
    test_dataset = RegressionDataset(
        x_data=torch.from_numpy(x_test).float(), y_data=torch.from_numpy(y_test).float()
    )

    # DataLoader

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        num_workers=cfg["DATASET"]["NUM_THREADS"],
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg["DATASET"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg["TEST"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"],
    )

    if cfg["DATASET"]["VERBOSITY"]:
        print(
            f"The train set contains {len(train_loader.dataset)} samples,"
            f" in {len(train_loader)} batches"
        )
        print(
            f"The validation set contains {len(valid_loader.dataset)} samples,"
            f" in {len(valid_loader)} batches"
        )
        print(
            f"The test set contains {len(test_loader.dataset)} samples,"
            f" in {len(test_loader)} batches"
        )

    return train_loader, valid_loader, test_loader


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

    main(cfg=config_file)
