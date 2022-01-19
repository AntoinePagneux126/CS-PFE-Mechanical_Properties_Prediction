"""This module aims to load and process the data."""
# pylint: disable=import-error, no-name-in-module
import argparse
import yaml

from dataset_utils import basic_random_split


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
    data = basic_random_split(path_to_data=path_to_data)

    return data


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
