# Configuration file

<div id="top"></div>
<br />
<div align="center">
  <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2/-/tree/master/src">
    <img src="../images/yaml_logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PFE ArcelorMittal</h3>

  <p align="center">
    We will explain in this part how to use the configuration file to train your models in the best possible way. It is divided into four major parts.
    <br />
    <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2"><strong> Back to main documentation »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#dataset-section">DATASET section</a>
      <ul>
        <li><a href="#select-files">Select Files</a></li>
        <li><a href="#preprocessing">Preprocessing</a></li>
      </ul>
    </li>
    <li>
      <a href="#models-section">MODELS section</a>
      <ul>
        <li><a href="#select-your-model-type">Select your model type</a></li>
        <li><a href="#parametrize-your-ml-algorithm">Parametrize your ML algorithm</a></li>
      </ul>
    </li>
    <li>
      <a href="#train-section">TRAIN section</a>
      <ul>
        <li><a href="#parametrize-your-nn-algorithm">Parametrize your NN algorithm</a></li>
      </ul>
    </li>
    <li>
      <a href="#test-section">TEST section</a>
    </li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## DATASET section

```yaml
DATASET:
  DATA_FORMAT: csv
  PREPROCESSING:
    MERGE_FILES:
      WHICH: "all"  # Either "all" or list of indexes. ex : [2] or [0, 1, 2] or [2, 0] ...
    NORMALIZE:
      TYPE: "StandardScaler"  # ["MinMaxScalar", "StandardScaler"]
    REMOVE_FEATURES:
      ACTIVE: True
      WHICH: ["Linespeed (m/min)", "B ppm"]
    REMOVE_SAMPLES:
      ACTIVE: False
      WHICH:
        #Direction: "T"  # ["T", "L"]
        Type: "JI5"  # ["JI5", "I20"]

    TARGET: "rm"  # ["rm", "re02", "A80"]
  BATCH_SIZE: 32
  TEST_VALID_RATIO: [0.1, 0.2]
  VERBOSITY: True
  NUM_THREADS: 4
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Select files

To select the files on which you want to train or on which you want to predict the target you must modify line 7 `DATASET/PREPROCESSING/MERGE_FILES/WHICH`.
<br />
You will have different options. Either you use all the file, or specific files.

| Number | File name |
|--------|-----------|
| all    | All files |
| 0      | Galma     |
| 1      | SDG3-v2   |
| 2      | SDG3.5    |
| 3      | EKO1      |
| 4      | SDG3      |
| 5      | Sagunto   |

<p align="right">(<a href="#top">back to top</a>)</p>

### Preprocessing

You will have access to different types of preprocessing.

* Normalize your data
  * StandardScaler: Standardize features by removing the mean and scaling to unit variance. [Find documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html);
  * MinMaxScaler: Transform features by scaling each feature to a given range. [Find documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html);
* Remove some features;
  * When this option is activated, you can specify on which features you will do your training;
  * Available features:
    | Direction<br> Type<br>Th mm<br> C ppm<br>Mn ppm<br> Si ppm<br>P ppm<br> S ppm<br>Al ppm<br> Ti ppm<br> Cr ppm |  Nb ppm<br>B ppm<br> Mo ppm<br>Linespeed (m/min)<br> SKP elongation (%)<br>Heating C/s 660 to 750<br> t_soaking_hot s<br>Cooling C/s<br>t_soaking_cold s<br>soaking_hot C<br> soaking_cold C |
    |-------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

* Remove samples;
  * To select more precise data on which to train, you can specify the type or the direction of the tensile test. <strong> Be careful, you fill in the samples you do not want to keep</strong>;
    * Direction: "T"  or "L";
    * Type: "JI5" or "I20".

<p align="right">(<a href="#top">back to top</a>)</p>

## MODELS section

```yaml
MODELS:
  NN: True
  ML:
    ACTIVE: False
    GRID_SEARCH: False
    TYPE: 'GradientBoosting'  # ["RandomForest", ExtraTrees, GradientBoosting]
    RandomForest:
      rm:
        bootstrap: False
        max_depth: 21
        max_features: 10
        min_samples_split: 6
        n_estimators: 100
        n_jobs: -1
      re02:
        bootstrap: False
        max_depth: 26
        max_features: 10
        min_samples_split: 6
        n_estimators: 110
        n_jobs: -1
      A80:
        bootstrap: False
        max_depth: 25
        max_features: 11
        min_samples_split: 4
        n_estimators: 105
        n_jobs: -1
    ExtraTrees:
      rm:
        bootstrap: False
        max_depth: 23
        max_features: 11
        min_samples_split: 6
        n_estimators: 105
        n_jobs: -1
      re02:
        bootstrap: False
        max_depth: 23
        max_features: 11
        min_samples_split: 6
        n_estimators: 110
        n_jobs: -1
      A80:
        bootstrap: False
        max_depth: 21
        max_features: 10
        min_samples_split: 6
        n_estimators: 70
        n_jobs: -1
    GradientBoosting:
      rm:
        max_depth: 23
        max_features: 11
        min_samples_split: 6
        n_estimators: 110
      re02:
        max_depth: 23
        max_features: 11
        min_samples_split: 6
        n_estimators: 110
      A80:
        max_depth: 21
        max_features: 10
        min_samples_split: 6
        n_estimators: 70
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Select your model type

The most important part of this section is the type of model you want to train. Either a neural network or a machine learning algorithm.

* Set `MODELS/NN` to `True` if you want to train a NN and see section `TRAIN` to parametrize it;
* Set `MODELS/NN` to `False` to use a machine learnig algorithm.

<p align="right">(<a href="#top">back to top</a>)</p>

### Parametrize your ML algorithm

* First option is to launch a GridSearch for the hyperparameters of your model;
* Then you can select which algorithm you want to use. For now, three algorithms have been implemented: RandomForest, ExtraTrees and GradientBoosting. If you want to add other models, go [here](https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2/-/blob/master/src/models/);
* Then if you have launch a GridSearch and you have new hyperparameters for a model for a target (rm, re02, A80), you can enter them in the `MODELS/ML` section.

<p align="right">(<a href="#top">back to top</a>)</p>

## TRAIN section

```yaml
TRAIN:
  EPOCH: 100
  CHECKPOINT_STEP: 20
  MODEL: LinearNet_3
  NUM_HIDDEN_NEURON: 64
  LOG_DIR: 'tensorboard/metrics'
  SAVE_DIR: '../models'
  LR:
    LR_INITIAL : 0.01
    TYPE: "ReduceLROnPlateau"  # [ReduceLROnPlateau, CyclicLR]
    ReduceLROnPlateau:
      LR_DECAY: 0.2
      LR_PATIENCE : 10
      LR_THRESHOLD : 1
    CyclicLR:
      BASE_LR: 0.00001
      MAX_LR: 0.001
      STEP_SIZE: 500
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Parametrize your NN algorithm

In this section, you can specify all the parameters for your NN.

* First you fill the number of epoch;
* The `CHECKPOINT_STEP` allows to save the model parameters every x epochs even if the model is not better (it allows to have some backup of the model performance evolution);
* One of the most important part of this section is to select the model you want to train.  For now, three models are available :
  * `LinearNet_1` : 1 hidden layer with `NUM_HIDDEN_NEURON` neurones;
  * `LinearNet_2` : 3 hidden layers;
  * `LinearNet_3` : 2 hidden layers with `NUM_HIDDEN_NEURON` and `NUM_HIDDEN_NEURON`/2
* If you want to create a new NN model, see [here](https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2/-/blob/master/src/models/);
* `NUM_HIDDEN_NEURON` is very important for `LinearNet_1` and `LinearNet_3`. It represent the number of neurones of your first hidden layer;
* All the results of your models will be save in `SAVE_DIR`. You don't need to change this value.
* Then you can choose between two learning schedulers, [ReduceOnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#:~:text=Reduce%20learning%20rate%20when%20a,the%20learning%20rate%20is%20reduced.) and [CyclicLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html);

<p align="right">(<a href="#top">back to top</a>)</p>

## TEST section

```yaml
TEST:
  BATCH_SIZE: 16
  SAVE_DIR: '../res'
  PATH_TO_MODEL: '../models/linearnet_3_0/best_model.pth'
  AVERAGE:
    ACTIVE: True
    PATH:
      - {MODEL: '../models/randomforest_23/model.pck', CONFIG: '../models/randomforest_23/config_file.yaml'}
      - {MODEL: '../models/gradientboosting_26/model.pck', CONFIG: '../models/gradientboosting_26/config_file.yaml'}
      - {MODEL: '../models/extratrees_23/model.pck', CONFIG: '../models/extratrees_23/config_file.yaml'}
```

Finally, the last section concerns inference. We can either load a model and test it on a dataset (see the DAATASET section to select the files) or load several models and average their predictions.

<p align="right">(<a href="#top">back to top</a>)</p>
