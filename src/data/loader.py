"""This module aims to load and process the data."""
# pylint: disable=import-error, no-name-in-module
import argparse
import torch
import yaml

from torch.utils.data import DataLoader
from data.dataset_utils import basic_random_split, merge_files, RegressionDataset


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
        preprocessing=cfg["DATASET"]["PREPROCESSING"],
        test_valid_ratio=cfg["DATASET"]["TEST_VALID_RATIO"],
        which=cfg["DATASET"]["PREPROCESSING"]["MERGE_FILES"]["WHICH"],
    )

    # Select the target
    target_to_predict = cfg["DATASET"]["PREPROCESSING"]["TARGET"]
    print(f"\nYou want to predict : {cfg['DATASET']['PREPROCESSING']['TARGET']}")

    x_train, y_train, x_valid, y_valid, x_test, y_test = merge_files(
        data=data, target_to_predict=target_to_predict
    )

    if not cfg["MODELS"]["NN"]:
        return (
            {
                "x_train": x_train,
                "y_train": y_train,
                "x_valid": x_valid,
                "y_valid": y_valid,
            },
            {"x_test": x_test, "y_test": y_test},
        )

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
