"""This module define functions to visualize different results."""
# pylint: disable=invalid-name
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_y_pred_y_true(
    y_true, y_pred, metrics, path_to_save, target_name: str = "$R_{m}$"
):
    """Plot y_pred with respect to y_true

    Args:
        y_true (array): Ground truth target values
        y_pred (array): Estimated target values
        metrics (dict): Metrics (RMSE, R2)
        path_to_save (str): Path to file
        target_name (str) : Name of the target, formated in TeX
    """
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(y_true, y_pred, "bo", label="Estimated target")
    ax.plot([min(y_true), max(y_true)], [min(y_pred), max(y_pred)], "r", label="y=x")

    ax.set_title(
        f"MLP: {target_name} estimated with respect to ground truth {target_name}",
        fontsize="large",
    )
    ax.set_xlabel("Ground truth target values")
    ax.set_ylabel("Estimated target values")

    ax.text(
        x=max(y_true),
        y=min(y_pred) + 30,
        s="R2=" + str(round(metrics["R2_test"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax.text(
        x=max(y_true),
        y=min(y_pred) + 15,
        s="MSE=" + str(round(metrics["MSE_test"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax.text(
        x=max(y_true),
        y=min(y_pred),
        s="RMSE=" + str(round(metrics["RMSE_test"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )

    ax.legend(loc="best")

    plt.savefig(os.path.join(path_to_save, "y_pred_y_true.png"))
    plt.show()


def plot_all_y_pred_y_true(
    y_true, y_pred, metrics: dict, path_to_save, target_name: str = "$R_{m}$"
):
    """Plot all y_preds with respect to y_trues

    Args:
        y_true (dict): Ground truth target values
        y_pred (dict): Estimated target values
        metrics (dict): Metrics (RMSE, R2)
        path_to_save (str): Path to file
        target_name (str) : Name of the target, formated in TeX
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    fig.tight_layout(pad=3)

    ax[0, 0].plot(y_true["train"], y_pred["train"], "bo", label="Estimated target")
    ax[0, 0].plot(
        [min(y_true["train"]), max(y_true["train"])],
        [min(y_pred["train"]), max(y_pred["train"])],
        "r",
        label="y=x",
    )

    ax[0, 0].set_title(
        f"Train: {target_name} estimated with respect to ground truth {target_name}",
        fontsize="large",
    )
    ax[0, 0].set_xlabel("Ground truth target values")
    ax[0, 0].set_ylabel("Estimated target values")
    ax[0, 0].text(
        x=max(y_true["train"]),
        y=min(y_pred["train"]) + 30,
        s="R2=" + str(round(metrics["R2_train"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0, 0].text(
        x=max(y_true["train"]),
        y=min(y_pred["train"]) + 15,
        s="MSE=" + str(round(metrics["MSE_train"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0, 0].text(
        x=max(y_true["train"]),
        y=min(y_pred["train"]),
        s="RMSE=" + str(round(metrics["RMSE_train"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0, 0].legend()

    ax[0, 1].plot(y_true["valid"], y_pred["valid"], "bo", label="Estimated target")
    ax[0, 1].plot(
        [min(y_true["valid"]), max(y_true["valid"])],
        [min(y_pred["valid"]), max(y_pred["valid"])],
        "r",
        label="y=x",
    )

    ax[0, 1].set_title(
        f"Valid: {target_name} estimated with respect to ground truth {target_name}",
        fontsize="large",
    )
    ax[0, 1].set_xlabel("Ground truth target values")
    ax[0, 1].set_ylabel("Estimated target values")
    ax[0, 1].text(
        x=max(y_true["valid"]),
        y=min(y_pred["valid"]) + 30,
        s="R2=" + str(round(metrics["R2_val"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0, 1].text(
        x=max(y_true["valid"]),
        y=min(y_pred["valid"]) + 15,
        s="MSE=" + str(round(metrics["MSE_val"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0, 1].text(
        x=max(y_true["valid"]),
        y=min(y_pred["valid"]),
        s="RMSE=" + str(round(metrics["RMSE_val"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0, 1].legend()

    ax[1, 0].plot(y_true["test"], y_pred["test"], "bo", label="Estimated target")
    ax[1, 0].plot(
        [min(y_true["test"]), max(y_true["test"])],
        [min(y_pred["test"]), max(y_pred["test"])],
        "r",
        label="y=x",
    )

    ax[1, 0].set_title(
        f"Test: {target_name} estimated with respect to ground truth {target_name}",
        fontsize="large",
    )
    ax[1, 0].set_xlabel("Ground truth target values")
    ax[1, 0].set_ylabel("Estimated target values")
    ax[1, 0].text(
        x=max(y_true["test"]),
        y=min(y_pred["test"]) + 30,
        s="R2=" + str(round(metrics["R2_test"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[1, 0].text(
        x=max(y_true["test"]),
        y=min(y_pred["test"]) + 15,
        s="MSE=" + str(round(metrics["MSE_test"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[1, 0].text(
        x=max(y_true["test"]),
        y=min(y_pred["test"]),
        s="RMSE=" + str(round(metrics["RMSE_test"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[1, 0].legend()

    ax[1, 1].plot(y_true["train"], y_pred["train"], "bo", label="Train")
    ax[1, 1].plot(y_true["valid"], y_pred["valid"], "ro", label="Valid")
    ax[1, 1].plot(y_true["test"], y_pred["test"], "go", label="Test")

    ax[1, 1].set_title(
        f"MLP: {target_name} estimated with respect to ground truth {target_name}",
        fontsize="large",
    )
    ax[1, 1].set_xlabel("Ground truth target values")
    ax[1, 1].set_ylabel("Estimated target values")
    ax[1, 1].legend()

    plt.savefig(os.path.join(path_to_save, "y_pred_y_true.png"))
    plt.show()


def plot_partial_y_pred_y_true(
    y_true, y_pred, metrics: dict, path_to_save, target_name: str = "$R_{m}$"
):
    """Plot partial y_preds with respect to y_trues

    Args:
        y_true (dict): Ground truth target values
        y_pred (dict): Estimated target values
        metrics (dict): Metrics (RMSE, R2)
        path_to_save (str): Path to file
        target_name (str) : Name of the target, formated in TeX
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    fig.tight_layout(pad=3)

    ax[0].plot(y_true["train"], y_pred["train"], "bo", label="Estimated target")
    ax[0].plot(
        [min(y_true["train"]), max(y_true["train"])],
        [min(y_pred["train"]), max(y_pred["train"])],
        "r",
        label="y=x",
    )

    ax[0].set_title(
        f"Train: {target_name} estimated with respect to ground truth {target_name}",
        fontsize="large",
    )
    ax[0].set_xlabel("Ground truth target values")
    ax[0].set_ylabel("Estimated target values")

    ax[0].text(
        x=max(y_true["train"]),
        y=min(y_pred["train"]) + 40,
        s="R2=" + str(round(metrics["R2_train"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0].text(
        x=max(y_true["train"]),
        y=min(y_pred["train"]) + 20,
        s="MSE=" + str(round(metrics["MSE_train"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0].text(
        x=max(y_true["train"]),
        y=min(y_pred["train"]),
        s="RMSE=" + str(round(metrics["RMSE_train"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[0].legend()

    ax[1].plot(y_true["valid"], y_pred["valid"], "bo", label="Estimated target")
    ax[1].plot(
        [min(y_true["valid"]), max(y_true["valid"])],
        [min(y_pred["valid"]), max(y_pred["valid"])],
        "r",
        label="y=x",
    )

    ax[1].set_title(
        f"Valid: {target_name} estimated with respect to ground truth {target_name}",
        fontsize="large",
    )
    ax[1].set_xlabel("Ground truth target values")
    ax[1].set_ylabel("Estimated target values")

    ax[1].text(
        x=max(y_true["valid"]),
        y=min(y_pred["valid"]) + 40,
        s="R2=" + str(round(metrics["R2_val"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[1].text(
        x=max(y_true["valid"]),
        y=min(y_pred["valid"]) + 20,
        s="MSE=" + str(round(metrics["MSE_val"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax[1].text(
        x=max(y_true["valid"]),
        y=min(y_pred["valid"]),
        s="RMSE=" + str(round(metrics["RMSE_val"], 3)),
        horizontalalignment="right",
        verticalalignment="center",
    )

    ax[1].legend()

    plt.savefig(os.path.join(path_to_save, "y_pred_y_true.png"))
    plt.show()


def plot_feature_importance(
    importance, names, model_type, path_to_save: str, target_name: str = "$R_{m}$"
):
    """Plot feature importance based on ML mpdel used

    Args:
        importance (list): feature importance values
        names (list): feature names
        model_type (string): Model used
        path_to_save (str): Path to file
        target_name (str) : Name of the target, formated in TeX
    """

    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Define size of bar plot
    fig = plt.figure(figsize=(18, 8))
    # Plot Searborn bar chart
    sns.barplot(
        x=fi_df["feature_importance"][0 : min(20, len(fi_df))],
        y=fi_df["feature_names"][0 : min(20, len(fi_df))],
    )
    # Add chart labels
    plt.title(f"{model_type} - Features Importance for {target_name} prediction")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")

    plt.savefig(path_to_save + "/feature_importance_" + model_type + ".png")
    plt.close(fig)


def plot_feature_distrib(
    importance, names, model_type, path_to_save: str, target_name: str = "$R_{m}$"
):
    """Plot feature importance distribution based on NN model used

    Args:
        importance (list): feature importance values
        names (list): feature names
        model_type (string): Model used
        path_to_save (str): Path to file
        target_name (str) : Name of the target, formated in TeX
    """

    fig = plt.figure(figsize=(14, 14))
    j = 0
    for i, name in enumerate(names):
        plt.subplot(5, 4, j + 1)
        j += 1
        sns.histplot(importance[:, i], label=name, kde=True, linewidth=0)
        plt.legend(loc="best")

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    # Add chart labels
    plt.suptitle(
        f"{model_type} - Features Importance distribution for {target_name} prediction"
    )

    plt.savefig(path_to_save + "/feature_importance_distrib_" + model_type + ".png")
    plt.close(fig)
