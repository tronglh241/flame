# flame

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch) ![GitHub](https://img.shields.io/github/license/tronglh241/flame) ![GitHub Repo stars](https://img.shields.io/github/stars/tronglh241/flame) ![GitHub forks](https://img.shields.io/github/forks/tronglh241/flame)

Flame is a library that helps develop neural networks fast and flexibly. It is built on PyTorch Ignite, a high-level library in PyTorch Ecosystem.

# Contents

- [Why flame?](#why-flame-)
- [Concepts](#concepts)
- [Installation](#installation)
- [Get started](#get-started)
    - [Usage](#usage)
    - [Run your first experiment](#run-your-first-experiment)

# Why flame?

When developing neural networks people train and evaluate models a lot and repeat these works on many problems. Flame is created for solving two needs:
- **Templates for doing experiments**: flame provides templates for neural network development with common utilities like saving checkpoints periodically, resume training, logging, evaluating, etc.
- **The way to add functionalities flexibly**: depending on different problems developers have different requirements for the training and testing. They might want to stop the training progress if there is no improvement, plotting the results after each epoch or they just want a vanilla training loop. Now with flame, you can use any on-the-shelf metrics and handlers from Ignite or your own just by editing the configuration file.

# Concepts
TBD
# Installation

Create your new environment with Python 3 and install flame by `pip`:
```bash
pip install pytorch-flame
```
# Get started

## Usage

Flame provides two commands:
- Initialize a new project
    ```
    usage: flame init [-h] [-f] [directory]

    positional arguments:
      directory   Directory in which the new project is initialized. If not specified, it will be initialized in the current directory.

    optional arguments:
      -h, --help  show this help message and exit
      -f, --full  Whether to create a full template project or not.
    ```
- Run the training or testing
    ```
    usage: flame run [-h] file

    positional arguments:
      file        Config file

    optional arguments:
      -h, --help  show this help message and exit
    ```

## Run your first experiment

Let's get started with a simple experiment: classification on the MNIST dataset.

1. Flame runs experiments with configs so you need to create configs first. Run commands
    ```bash
    mkdir mnist-classifcation && cd mnist-classification
    flame init
    ```
    or you can run just command
    ```bash
    flame init mnist-classification
    ```
    flame will create the folder and initialize in it. The folder created will have the structure:
    ```
    mnist-classification/
    └── configs
        ├── test.yml
        └── train.yml
    ```
    You can add `-f` or `--full` to `init` command for creating an extra file `run.py` in case you prefer running `python run.py` rather than `flame run` for some reason. Then the structure will be:
    ```
    mnist-classification/
    ├── configs
    │   ├── test.yml
    │   └── train.yml
    └── run.py
    ```
2. MNIST dataset and the model will be got from `torchvision`, so we need to install it.
    ```bash
    pip install torchvision
    ```
3. Now, you have all for the training. `cd` to `mnist-classification` and run it by
    ```bash
    flame run configs/train.yml
    ```
    To see how the training is going on, start Tensorboard
    ```bash
    tensorboard --logdir logs/
    ```
4. Checkpoints will be saved in `checkpoints` folder. Say the training is done and you want to evaluate the model `checkpoints/best_model.pt`, for example, change value `checkpoint.loader.kwargs.path` in `configs/test.yml` to `checkpoints/best_model.pt`.
    ```yaml
    checkpoint:
      loader:
        module: flame.handlers
        name: CheckpointLoader
        kwargs:
          path: "'checkpoints/best_model.pt'"
    ```
    Run the following command to start evaluating the model:
    ```bash
    flame run configs/test.yml
    ```

That's it! You have just completed training and evaluating with flame.
