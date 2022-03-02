"""This module define functions to visualize different results."""
# pylint: disable=invalid-name
import os
import matplotlib.pyplot as plt


def plot_y_pred_y_true(y_true, y_pred, path_to_save, target_name: str = "$R_{m}$"):
    """Plot y_pred with respect to y_true

    Args:
        y_true (array): Ground truth target values
        y_pred (array): Estimated target values
        path_to_save (str): Path to file
    """
    fig = plt.figure(figsize=(8, 15))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(y_true, y_pred, "bo", label="Estimated target")
    ax.plot([min(y_true), max(y_true)], [min(y_pred), max(y_pred)], "r", label="y=x")

    ax.set_title(
        f"MLP: {target_name} estimated with respect to ground truth {target_name}",
        fontsize="large",
    )
    ax.set_xlabel("Ground truth target values")
    ax.set_ylabel("Estimated target values")
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
