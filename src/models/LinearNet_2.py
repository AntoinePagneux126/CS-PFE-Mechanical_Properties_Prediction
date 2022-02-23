# pylint: disable=invalid-name
"""Create a Linear NN."""
import torch
import torch.nn as nn


class LinearNet_2(
    nn.Module
):  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """Define our Linear model"""

    def __init__(self, num_features):
        super(LinearNet_2, self).__init__()  # pylint: disable=super-with-arguments
        # Couche d'entrée vers la couche cachée

        self.lin1 = nn.Linear(num_features, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 16)
        self.lin4 = nn.Linear(16, 1)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(16)
        self.drops = nn.Dropout(0.1)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

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

        x = self.leaky_relu(self.lin1(inputs))
        x = self.drops(x)
        x = self.bn2(x)
        x = self.leaky_relu(self.lin2(x))
        # x = self.drops(x)
        x = self.bn3(x)
        x = self.leaky_relu(self.lin3(x))
        x = self.drops(x)
        x = self.bn4(x)
        x = self.lin4(x)
        return x
