from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import ignite.engine as e
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Engine(e.Engine, ABC):
    def __init__(self, model: Module, data: DataLoader, device: Any = 'cpu', max_epochs: int = None,
                 epoch_length: int = None):
        super(Engine, self).__init__(self.process)
        self.model = model
        self.data = data
        self.device = device
        self.max_epochs = max_epochs
        self.epoch_length = epoch_length

        self.model.to(self.device)

    def run(self) -> e.State:
        return super(Engine, self).run(self.data, self.max_epochs, self.epoch_length)

    @abstractmethod
    def process(self, engine: e.Engine, batch: Iterable) -> Any:
        pass


class Trainer(Engine):
    def __init__(self, loss: Callable, optim: Optimizer, **kwargs: Any):
        super(Trainer, self).__init__(**kwargs)
        self.loss = loss
        self.optim = optim

    def process(self, engine: e.Engine, batch: Iterable) -> Any:
        self.model.train()
        self.optim.zero_grad()
        params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
        params[0] = self.model(params[0])
        loss = self.loss(*params)
        loss.backward()
        self.optim.step()
        return loss.item()


class Evaluator(Engine):
    def process(self, engine: e.Engine, batch: Iterable) -> Any:
        self.model.eval()
        with torch.no_grad():
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            params[0] = self.model(params[0])
        return params
