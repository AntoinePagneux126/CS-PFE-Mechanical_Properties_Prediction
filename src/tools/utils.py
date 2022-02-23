"""This module aims to define utils function for the project."""
from torch.optim import lr_scheduler

from models.LinearNet_1 import LinearNet_1
from models.LinearNet_2 import LinearNet_2
from models.LinearNet_3 import LinearNet_3


def load_model(cfg, input_size, num_hidden_neuron):
    """This function aims to load the right model regarding the configuration file

    Args:
        cfg (dict): Configuration file

    Returns:
        nn.Module: Neural Network
    """
    if cfg["TRAIN"]["MODEL"] == "LinearNet_1":
        return LinearNet_1(num_features=input_size, num_hidden_neuron=num_hidden_neuron)
    elif cfg["TRAIN"]["MODEL"] == "LinearNet_2":
        return LinearNet_2(num_features=input_size)
    elif cfg["TRAIN"]["MODEL"] == "LinearNet_3":
        return LinearNet_3(num_features=input_size, num_hidden_neuron=num_hidden_neuron)
    else:
        return LinearNet_1(num_features=input_size, num_hidden_neuron=num_hidden_neuron)


def choose_scheduler(cfg, optimizer):
    """This function aims to choose between different learning rate scheduler

    Args:
        cfg (dict): Configuration file

    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler
    """
    if cfg["TYPE"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg["ReduceLROnPlateau"]["LR_DECAY"],
            patience=cfg["ReduceLROnPlateau"]["LR_PATIENCE"],
            threshold=cfg["ReduceLROnPlateau"]["LR_THRESHOLD"],
        )
    elif cfg["TYPE"] == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg["CyclicLR"]["BASE_LR"],
            max_lr=cfg["CyclicLR"]["MAX_LR"],
            step_size_up=cfg["CyclicLR"]["STEP_SIZE"],
            cycle_momentum=False,
        )
    return scheduler
