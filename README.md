<div id="top"></div>
<br />
<div align="center">
  <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2">
    <img src="images/logo.png" alt="Logo" width="160" height="80">
  </a>

  <h3 align="center">PFE ArcelorMittal</h3>

  <p align="center">
    Analysis of a very high strength steel manufacturing line using ML and DL methods for the prediction of mechanical properties.
    <br />
    <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2"><strong>Explore the docs »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#navigate-into-the-project">Navigate into the project</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The objective of this project is to analyze a manufacturing line of very high strength steel (VHS) by machine learning methods for the prediction of the mechanical qualities of these steels. Very high strength steels are very interesting in the industry because of their ability to combine antagonistic mechanical properties such as mechanical strength and ductility. The following definitions can be given:

* Ductility : Ability of a material to deform plastically without breaking. It is generally characterized by the elongation at break A%. The higher the A%, the more ductile the material.
* Mechanical strength: Ability of a material to resist a mechanical stress without breaking. It is generally noted $R_m$.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

This project has been mainly developed in python using standard machine learning and deep learning libraries such as the pytorch and sklearn frameworks.

* [Python](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

For more details on the modules and libraries used, please refer to the [requirement.txt](https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2/-/blob/master/requirements.txt) file

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Navigate into the project

```bash
.
├── README.md
├── requirement.txt
├── data
│   ├── DP980_Sagunto_2021_upgrade.csv
│   ├── DP980GA_Galma1_upgrade.csv
│   ├── DP980GI_SDG3-v2_20220208_upgrade.csv
│   ├── dDP980GI_SDG3.5_20220208_upgrade.csv
│   ├── DP980Y700_EKO1_upgrade.csv
│   └── DP980Y700_SDG3_2021_upgrade.csv
├── docs
├── images
├── models
├── notebooks
├── res
└── src
    ├── README.MD
    ├── average_inference.py
    ├── config.yaml
    ├── inference.py
    ├── train.py
    ├── data
    │   ├── loader.py
    │   └── dataset_utils.py
    ├── models
    │   ├── LinearNet_1.py
    │   ├── LinearNet_2.py
    │   ├── LinearNet_3.py
    │   └── MachineLearningModels.py
    ├── tensorboard
    ├── tools
    │   ├── trainer.py
    │   ├── utils.py
    │   └── valid.py
    └── visualization
        └── vis.py
```

* `./data`: Contains all data files
* `./docs`: Contains all the files concerning the documentation of the code
* `./models`: Contains all trained models. This is the output directory
* `./notebooks`: Contains all the notebooks to speed up development and test new things
* `./src`: Contains all the scripts of the project

<p align="right">(<a href="#top">back to top</a>)</p>

### Prerequisites

You will need Python on your computer and pip (the package installer for Python).

### Installation

1. Clone the repo

   ```sh
   git clone https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2.git
   ```

2. Prepare your virtual environment

   ```sh
   pip install virtualenv
   python3 -m venv pfe_arcelormittal
   source pfe_arcelormittal/bin/activate
   pip --version
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

**All our project is configurable using the configuration file in the `src` folder. For more details about this configuration file, please refer to our specific documentation [[here]](https://gitlab-student.centralesupelec.fr/2018barreeg/pfe-arcelor2/-/blob/master/src/README.md)**

### Launch the training

```bash
cd ./src
python3 train.py --path_to_config ./config.yaml
```

### Launch inference on the test set

```bash
cd ./src
python3 inference.py --path_to_config ./config.yaml
```

### Launch model averaging on the test set

```bash
cd ./src
python3 average_inference.py --path_to_config ./config.yaml
```

### Generate the documentation

```bash
cd ./docs
sphinx-apidoc -o ./source ../src
make html

cd ./build/html
firefox index.html
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Barrée Guillaume - guillaume.barree@student-cs.fr
<br />
Pagneux Antoine - antoine.pagneux@student-cs.fr

<p align="right">(<a href="#top">back to top</a>)</p>
