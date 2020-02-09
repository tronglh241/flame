from ...module import Module


class TorchOptim(Module):
    def __init__(self, optim, optim_kwargs):
        super(TorchOptim, self).__init__()
        self.optim = optim
        self.optim_kwargs = optim_kwargs

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        optim = self.optim(self.frame['model'].parameters(), **self.optim_kwargs)
        self.frame[self.module_name] = optim