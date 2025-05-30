from typing import Any, Callable, Union

import torch
from ignite.engine import Engine as _Engine
from torch.nn import Module
from torch.utils.data import DataLoader


class Engine(_Engine):
    '''
    A custom extension of Ignite's Engine that binds a model and dataloader for training or evaluation.

    Args:
        data (DataLoader): The data loader that provides batches of input data.
        model (Module): The PyTorch model to be trained or evaluated.
        process_function (Callable[[Engine, Any], Any]): A function that processes one batch of data.
        device (Union[str, torch.device], optional): Device on which the model runs (e.g., 'cpu', 'cuda').
        max_epochs (int, optional): Maximum number of epochs to run. If None, defaults to 1.
        epoch_length (int, optional): Number of iterations per epoch. If None, defaults to len(data).
    '''

    def __init__(
        self,
        data: DataLoader,
        model: Module,
        process_function: Callable[[_Engine, Any], Any],
        device: Union[str, torch.device] = None,
        max_epochs: int = None,
        epoch_length: int = None,
    ):
        super(Engine, self).__init__(process_function)
        self.data = data
        self.model = model
        self.device = device
        self.max_epochs = max_epochs
        self.epoch_length = epoch_length

    def run(self) -> None:
        super(Engine, self).run(
            data=self.data,
            max_epochs=self.max_epochs,
            epoch_length=self.epoch_length,
        )
