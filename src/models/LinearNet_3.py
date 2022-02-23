# pylint: disable=invalid-name
"""Create a Linear NN."""
import torch
import torch.nn as nn


class LinearNet_3(
    nn.Module
):  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """Define our Linear model"""

    def __init__(self, num_features, num_hidden_neuron):
        super(LinearNet_3, self).__init__()  # pylint: disable=super-with-arguments
        # Couche d'entrée vers la couche cachée
        self.layer_1 = nn.Linear(num_features, num_hidden_neuron)
        self.layer_2 = nn.Linear(num_hidden_neuron, int(num_hidden_neuron / 2))
        # Cocuhe cachée vers la couche de sortie
        self.layer_out = nn.Linear(int(num_hidden_neuron / 2), 1)

        self.bn1 = nn.BatchNorm1d(num_hidden_neuron)
        self.bn2 = nn.BatchNorm1d(int(num_hidden_neuron / 2))
        self.drops = nn.Dropout(0.1)

    #  Méthode appelée lors de l'apprentissage du PMC.
    # Les fonctions d'activation des neurones de la couche
    # cachée sont des sigmoïd. Le neurone de la couche de sortie
    #  possède une fonction d'activation linéaire
    def forward(self, inputs):
        """Define the forward method

        Args:
            inputs (torch.Tensor): input data

        Returns:
            torch.Tensor: labels
        """

        x = torch.sigmoid(self.layer_1(inputs))
        x = self.bn1(x)
        x = torch.sigmoid(self.layer_2(x))
        x = self.bn2(x)
        x = self.layer_out(x)
        return x
