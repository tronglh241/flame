from torch import nn
from ...module import Module


class Model(nn.Module, Module):
    def __init__(self):
        super(Model, self).__init__()

    def init(self):
        pass
