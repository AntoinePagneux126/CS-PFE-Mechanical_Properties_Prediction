"""This file contains all functions related to the dataset."""
# pylint: disable=import-error
import os
import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    """Create a Torch Dataset for our regression problem."""

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)


def basic_random_split(
    path_to_data, preprocessing, test_valid_ratio=(0.1, 0.2), which="all"
):
    """This function split file according to a ratio to create
    training, validation and test dataset.

    Args:
        path_to_data (str): path of the data root directory.
        valid_ratio (tuple(float, float)): ratio of data for test and validation dataset.

    Returns:
        dict: Dictionary containing every data to create a Dataset.
    """
    # Load the different files
    data_from_lines = load_files(path_to_data=path_to_data, which=which)

    # Transform qualitative informations into quantitative ones
    data_from_lines = handle_qualitative_data(
        data_from_lines=data_from_lines, preprocessing=preprocessing
    )

    # Prepare features and targets
    features_and_targets = remove_useless_features(
        data_from_lines=data_from_lines, preprocessing=preprocessing
    )

    for key in features_and_targets:
        features_and_targets[key]["dataset"] = create_x_and_y(
            input_data=features_and_targets[key], test_valid_ratio=test_valid_ratio
        )

    for key in features_and_targets:
        features_and_targets[key]["dataset"] = apply_preprocessing(
            data=features_and_targets[key]["dataset"],
            type_=preprocessing["NORMALIZE"]["TYPE"],
        )

    return features_and_targets


def load_files(path_to_data, which):
    """Load data from different file.

    Args:
        path_to_data (str): path of the data root directory.
        which (str or list): list of files we should load

    Returns:
        list(pandas.core.frame.DataFrame): List of Dataframe containing data from each file.
    """
    data = []
    data_files = np.array(sorted(os.listdir(path_to_data)))

    # Verify if which is type list or str
    which = [which] if isinstance(which, int) else which
    # Select files to load
    index_to_keep = (
        np.arange(len(data_files))
        if which == "all"
        else set(which).intersection(np.arange(len(data_files)))
    )

    if len(data_files) != 0:
        print("\n#################")
        print("# Loading files #")
        print("#################\n")
        print(f"You selected : {data_files[list(index_to_keep)]}\n")
        for datafile in tqdm.tqdm(data_files[list(index_to_keep)]):
            data.append(
                pd.read_csv(
                    os.path.join(path_to_data, datafile), delimiter=";", decimal=","
                )
            )

    return data


def handle_qualitative_data(data_from_lines, preprocessing):
    """Transform qualitative data into quantitative ones.

    Args:
        data_from_lines (list): List of Dataframe containing data from each file.
        preprocessing (dict): What type of preprocessing we want to apply (remove samples).

    Returns:
        list(pandas.core.frame.DataFrame): List of Dataframe containing data from each file.
    """
    has_print = False
    removed_values = ""
    cleanup = {"Direction": {"L": 1, "T": -1}, "Type": {"JI5": -1, "I20": 1}}

    for data in data_from_lines:
        # Remove samples
        if preprocessing["REMOVE_SAMPLES"]["ACTIVE"]:
            for feature in preprocessing["REMOVE_SAMPLES"]["WHICH"]:
                data.drop(
                    np.where(
                        data[feature]
                        == preprocessing["REMOVE_SAMPLES"]["WHICH"][feature]
                    )[0],
                    axis=0,
                    inplace=True,
                )
                if not has_print:
                    removed_values += f'\t{feature} : {preprocessing["REMOVE_SAMPLES"]["WHICH"][feature]}\n'  # pylint: disable=line-too-long
        data.replace(cleanup, inplace=True)
        has_print = True

    if removed_values:
        print("\n##################")
        print("# Remove samples #")
        print("##################\n")
        print(f"You removed : \n{removed_values}")
    return data_from_lines


def remove_useless_features(data_from_lines, preprocessing):
    """Create features and targets

    Args:
        data_from_lines (list): List of Dataframe containing data from each file.
        preprocessing (dict): What type of preprocessing we want to apply
                                (remove features for example).

    Returns:
        dict : Dictionary containing features and target for each file.
    """
    data_dict = {}
    to_be_removed = ["Coilnr", "Date"]
    if preprocessing["REMOVE_FEATURES"]["ACTIVE"]:
        for feature in preprocessing["REMOVE_FEATURES"]["WHICH"]:
            to_be_removed += [feature] if feature in data_from_lines[0].columns else []

        print("\n###################")
        print("# Remove features #")
        print("###################\n")
        print(f"You removed : \n\t{to_be_removed}")

    for i, data in enumerate(data_from_lines):
        if data.empty:
            continue

        data.reset_index(inplace=True)

        filters = data[to_be_removed]
        target_re02 = data[["Re02 Mpa"]]
        target_rm = data[["Rm Mpa"]]
        target_a = data[["A80 x10%"]]

        features = data.drop(target_a + target_re02 + target_rm + filters, axis=1)

        data_dict[i] = {
            "features": features,
            "re02": target_re02,
            "rm": target_rm,
            "A80": target_a,
        }
    return data_dict


def create_x_and_y(input_data, test_valid_ratio):  # pylint: disable=too-many-locals
    """Generate train, valid and test for each file and for each target.

    Args:
        input_data (dict): Features and targets for one file.
        test_valid_ratio (list(float, float)): Test and validation ratio.

    Returns:
        dict: train, valid and test inputs and targets.
    """
    feature_and_target = {}

    # Compute for rm
    x_train_valid_rm, x_test_rm, y_train_valid_rm, y_test_rm = train_test_split(
        input_data["features"],
        input_data["rm"],
        test_size=test_valid_ratio[0],
        random_state=0,
    )
    x_train_rm, x_valid_rm, y_train_rm, y_valid_rm = train_test_split(
        x_train_valid_rm,
        y_train_valid_rm,
        test_size=test_valid_ratio[1],
        random_state=0,
    )
    y_train_rm = y_train_rm.values.ravel()
    y_valid_rm = y_valid_rm.values.ravel()
    y_test_rm = y_test_rm.values.ravel()

    feature_and_target["rm"] = {
        "x_train": x_train_rm.to_numpy(),
        "y_train": y_train_rm,
        "x_valid": x_valid_rm.to_numpy(),
        "y_valid": y_valid_rm,
        "x_test": x_test_rm.to_numpy(),
        "y_test": y_test_rm,
    }

    # Compute for re
    x_train_valid_re02, x_test_re02, y_train_valid_re02, y_test_re02 = train_test_split(
        input_data["features"],
        input_data["re02"],
        test_size=test_valid_ratio[0],
        random_state=0,
    )
    x_train_re02, x_valid_re02, y_train_re02, y_valid_re02 = train_test_split(
        x_train_valid_re02,
        y_train_valid_re02,
        test_size=test_valid_ratio[0],
        random_state=0,
    )
    y_train_re02 = y_train_re02.values.ravel()
    y_valid_re02 = y_valid_re02.values.ravel()
    y_test_re02 = y_test_re02.values.ravel()

    feature_and_target["re02"] = {
        "x_train": x_train_re02.to_numpy(),
        "y_train": y_train_re02,
        "x_valid": x_valid_re02.to_numpy(),
        "y_valid": y_valid_re02,
        "x_test": x_test_re02.to_numpy(),
        "y_test": y_test_re02,
    }

    # Compute for A80
    x_train_valid_a80, x_test_a80, y_train_valid_a80, y_test_a80 = train_test_split(
        input_data["features"],
        input_data["A80"],
        test_size=test_valid_ratio[0],
        random_state=0,
    )
    x_train_a80, x_valid_a80, y_train_a80, y_valid_a80 = train_test_split(
        x_train_valid_a80,
        y_train_valid_a80,
        test_size=test_valid_ratio[0],
        random_state=0,
    )
    y_train_a80 = y_train_a80.values.ravel()
    y_valid_a80 = y_valid_a80.values.ravel()
    y_test_a80 = y_test_a80.values.ravel()

    feature_and_target["A80"] = {
        "x_train": x_train_a80.to_numpy(),
        "y_train": y_train_a80,
        "x_valid": x_valid_a80.to_numpy(),
        "y_valid": y_valid_a80,
        "x_test": x_test_a80.to_numpy(),
        "y_test": y_test_a80,
    }

    return feature_and_target


def apply_preprocessing(data, type_):
    """Normalize the data

    Args:
        data (dict:) train, valid and test inputs and targets.
        type_ (str): Type of normalization to apply

    Returns:
        dict: Normalized data
    """
    scaler = MinMaxScaler() if type_ == "MinMaxScalar" else StandardScaler()

    # rm
    data["rm"]["x_train"] = scaler.fit_transform(data["rm"]["x_train"])
    data["rm"]["x_valid"] = scaler.transform(data["rm"]["x_valid"])
    data["rm"]["x_test"] = scaler.transform(data["rm"]["x_test"])

    # re02
    data["re02"]["x_train"] = scaler.fit_transform(data["re02"]["x_train"])
    data["re02"]["x_valid"] = scaler.transform(data["re02"]["x_valid"])
    data["re02"]["x_test"] = scaler.transform(data["re02"]["x_test"])

    # A80
    data["A80"]["x_train"] = scaler.fit_transform(data["A80"]["x_train"])
    data["A80"]["x_valid"] = scaler.transform(data["A80"]["x_valid"])
    data["A80"]["x_test"] = scaler.transform(data["A80"]["x_test"])

    return data


def merge_files(data, target_to_predict):
    """Merge all files to create on dataset composed of different production lignes.

    Args:
        data (dict): Dictionary containing train, valid, test set for all lignes
        target_to_predict (str): Either rm, re02 or A80

    Returns:
        (array, ...): x_train, y_train, x_valid, y_valid, x_test, y_test
    """
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    x_test, y_test = None, None

    for key in data.keys():
        # Train
        x_train = (
            np.concatenate(
                (x_train, data[key]["dataset"][target_to_predict]["x_train"]), axis=0
            )
            if x_train is not None
            else data[key]["dataset"][target_to_predict]["x_train"]
        )
        y_train = (
            np.concatenate(
                (y_train, data[key]["dataset"][target_to_predict]["y_train"]), axis=0
            )
            if y_train is not None
            else data[key]["dataset"][target_to_predict]["y_train"]
        )
        # Valid
        x_valid = (
            np.concatenate(
                (x_valid, data[key]["dataset"][target_to_predict]["x_valid"]), axis=0
            )
            if x_valid is not None
            else data[key]["dataset"][target_to_predict]["x_valid"]
        )
        y_valid = (
            np.concatenate(
                (y_valid, data[key]["dataset"][target_to_predict]["y_valid"]), axis=0
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

    return x_train, y_train, x_valid, y_valid, x_test, y_test
