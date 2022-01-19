"""This module aims to define utils function for the project."""
from models.LinearNet_1 import LinearNet_1


def load_model(cfg, input_size, num_hidden_neuron):
    """This function aims to load the right model regarding the configuration file

    Args:
        cfg (dict): Configuration file

    Returns:
        nn.Module: Neural Network
    """
    if cfg["TRAIN"]["MODEL"] == "LinearNet_1":
        return LinearNet_1(num_features=input_size, num_hidden_neuron=num_hidden_neuron)
    else:
        return LinearNet_1(num_features=input_size, num_hidden_neuron=num_hidden_neuron)
