# pylint: disable=invalid-name
"""Create a Linear NN."""
import torch
import torch.nn as nn


class LinearNet_1(nn.Module):  # pylint: disable=too-few-public-methods
    """Define our Linear model"""

    def __init__(self, num_features, num_hidden_neuron):
        super(LinearNet_1, self).__init__()  # pylint: disable=super-with-arguments
        # Couche d'entrée vers la couche cachée
        self.layer_1 = nn.Linear(num_features, num_hidden_neuron)
        # Cocuhe cachée vers la couche de sortie
        self.layer_out = nn.Linear(num_hidden_neuron, 1)

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
        x = self.layer_out(x)
        return x
