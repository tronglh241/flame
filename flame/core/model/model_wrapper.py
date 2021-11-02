from typing import Any, Callable

from torch.nn import Module


class ModelWrapper:
    def __init__(
        self,
        model: Module,
        input_transform: Callable = lambda x: x,
    ):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.input_transform = input_transform

    def __call__(self, *args) -> Any:
        args = self.input_transform(args)
        output = self.model(*args)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)
