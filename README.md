# flame

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch) ![GitHub](https://img.shields.io/github/license/tronglh241/flame) ![GitHub Repo stars](https://img.shields.io/github/stars/tronglh241/flame) ![GitHub forks](https://img.shields.io/github/forks/tronglh241/flame) [![Tests](https://github.com/tronglh241/flame/actions/workflows/tests.yml/badge.svg)](https://github.com/tronglh241/flame/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/tronglh241/flame/branch/master/graph/badge.svg?token=FF3UAKNLPF)](https://codecov.io/gh/tronglh241/flame)

**Flame** is a lightweight and efficient library for developing neural networks, built using PyTorch Ignite. It simplifies the training, evaluation, and experimentation processes by providing reusable templates and a highly configurable setup. With Flame, you can easily manage everything from logging and checkpoints to custom training loops, making it ideal for both beginners and experienced researchers looking to streamline their workflow.

## Contents

- [Why Use Flame?](#why-use-flame)
- [Concepts](#concepts)
- [Installation](#installation)
- [Getting Started](#getting-started)

## Why Use Flame?

Flame addresses two common needs in neural network development:

- **Experiment Templates**: Provides templates for training and evaluation with utilities like checkpoint saving, training resumption, logging, and more.
- **Flexible Functionality**: Easily customize training and testing by editing configuration files. Integrate with Ignite's metrics and handlers or use your own.

## Concepts

Details coming soon.

## Installation

Set up a Python 3 environment and install Flame using `pip`:

```bash
pip install git+https://github.com/tronglh241/flame.git
```

## Getting Started

### Usage

Flame includes two commands:

- **Initialize a new project**:
    ```bash
    flame init [-h] [-f] [directory]
    ```
    - `directory`: Location to initialize the project. Defaults to the current directory.
    - `-f, --full`: Creates a full project template with an additional `run.py` file.

- **Run training or testing**:
    ```bash
    flame run [-h] file
    ```
    - `file`: Path to the config file.

### Run Your First Experiment

Follow these steps to run a simple classification experiment using the MNIST dataset:

1. **Initialize a project**:
    ```bash
    mkdir mnist-classification && cd mnist-classification
    flame init
    ```
    Or, run directly:
    ```bash
    flame init mnist-classification
    ```
    The folder structure will look like this:
    ```
    mnist-classification/
    └── configs
        ├── test_ddp.yml
        ├── test.yml
        ├── train_ddp.yml
        └── train.yml
    ```
    Use `-f` for an extra `run.py` file:
    ```
    mnist-classification/
    ├── configs
    │   ├── test_ddp.yml
    │   ├── test.yml
    │   ├── train_ddp.yml
    │   └── train.yml
    └── run.py
    ```

2. **Install additional dependencies**:
    ```bash
    pip install torchvision
    ```

3. **Run the training**:
    ```bash
    flame run configs/train.yml
    ```
    To monitor training, use Tensorboard:
    ```bash
    tensorboard --logdir logs/
    ```

4. **Evaluate the model**:
    Update `checkpoint.loader.kwargs.path` in `configs/test.yml` to point to the trained model, e.g., `checkpoints/best_model.pt`:
    ```yaml
    checkpoint:
      loader:
        module: flame.handlers
        name: CheckpointLoader
        kwargs:
          path: "'checkpoints/best_model.pt'"
    ```
    Run the evaluation:
    ```bash
    flame run configs/test.yml
    ```

That's it! You've successfully trained and evaluated a model using Flame.
