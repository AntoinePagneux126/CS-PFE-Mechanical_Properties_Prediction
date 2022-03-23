# Source models

<div id="top"></div>
<br />
<div align="center">
  <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2/-/tree/master/src">
    <img src="../../images/AI_logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PFE ArcelorMittal</h3>

  <p align="center">
    We will explain in this part how you can add a ML/NN model.
    <br />
    <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2"><strong> Back to main documentation »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#create-a-new-model">Create a new model</a>
      <ul>
        <li><a href="#neural-network">Neural Network</a></li>
        <li><a href="#machine-learning-algorithm">Machine Learning algorithm</a></li>
      </ul>
    </li>
    <li>
      <a href="#add-your-model-to-the-pipeline">Add your model to the pipeline</a>
    </li>
  </ol>
</details>

## Create a new model

### Neural Network

```python
# pylint: disable=invalid-name
"""Create a Linear NN."""
import torch
import torch.nn as nn

class NeuralNetworkModel(nn.Module):  # pylint: disable=too-few-public-methods
    """Define our Linear model"""

    def __init__(self, num_features, num_hidden_neuron):
        super(NeuralNetworkModel, self).__init__()  # pylint: disable=super-with-arguments

        # Couche d'entrée vers la couche cachée
        self.layer_1 = nn.Linear(num_features, num_hidden_neuron)
        # Cocuhe cachée vers la couche de sortie
        self.layer_out = nn.Linear(num_hidden_neuron, 1)

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
```

Here is the basic code to create a NN

* Create a new python module an copy the code above;
* Change the name of the class (do not forget to put the same name in the `super` function);
* Define your different layers in the `__init__` method;
* Implement your `forward` method.

**Go to the last section to add your module to the pipeline** <a href="#add-your-model-to-the-pipeline">[to section]</a>

### Machine Learning algorithm

* You just have to add an `elif` statement in the models function of the `MachineLearningModels` module;
* Do not forget to add your model in the configuration file. See [[here]](https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2/-/blob/master/src/README.m#L116/).

**Go to the last section to add your module to the pipeline** <a href="#add-your-model-to-the-pipeline">[to section]</a>

## Add your model to the pipeline

The next step to select your model for training is to add it to the pipeline.

* Go to `./src/tools/utils.py` [[here]](https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2/-/blob/master/src/tools/utils.py)
* If your create a new module with a NN
  * Import it;
  * In `load_model` function, add an `elif` statement to be able to choose your model
* If you add an ML model
  * Your model will be automatically added to the pipeline if you did the step discribe in the section `Machine Learning algorithm` above;
  * In the function `launch_grid_search`, add an `elif` statement to be able to launch a grid search with your new model.
