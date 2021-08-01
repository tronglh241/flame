# flamev2

flamev2 is a [PyTorch](https://pytorch.org/) training template to help you train and evaluate neural networks flexibly.


## Installation

Use the package manager [conda](https://www.anaconda.com/) to setup the environment for flamev2.

```
$ conda create -n <env_name> -c conda-forge -c pytorch --file requirements.txt
```

## Usage
```
$ python -m flame -h
usage: __main__.py [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
                   [--engine ENGINE] [--model MODEL]
                   [--checkpointer CHECKPOINTER] [--setup SETUP]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -f CONFIG
  --checkpoint CHECKPOINT, -p CHECKPOINT
  --engine ENGINE
  --model MODEL
  --checkpointer CHECKPOINTER
  --setup SETUP
```
Basic usage:
- Training: ```$ python -m flame -f <config_path>```
- Testing or training with pretrained model: ```$ python -m flame -f <config_path> -p <model_path>```
- Resuming: ```$ python -m flame -p <backup_checkpoint_path>```

Example:
```
$ python -m flame -f configs/mnist_training.yaml  # Training
$ python -m flame -f configs/mnist_training.yaml -p best_model_10_loss\=-0.0386.pt  # Retraining
$ python -m flame -f configs/mnist_test.yaml -p best_model_10_loss\=-0.0386.pt  # Testing
$ python -m flame -p backup_checkpoint_15.pt  # Resuming
```
Run **Tensorboard** to monitor training progress: ```tensorboard --logdir <log_dir>```
