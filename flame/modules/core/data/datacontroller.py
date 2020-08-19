from .... import utils
from ...module import Module


class DataController(Module):
    def __init__(self, **configs):
        super(DataController, self).__init__()
        self.configs = configs
        self.dataloaders = {}

    def init(self):
        for loader_type, loader_configs in self.configs.items():
            self.dataloaders[loader_type] = utils.create_dataloader(loader_configs)

    def __call__(self, loader_type):
        return self.dataloaders[loader_type]
