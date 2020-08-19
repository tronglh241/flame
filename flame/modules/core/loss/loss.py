from ...module import Module


class LossBase(Module):
    def __init__(self, output_transform=lambda x: x):
        super(LossBase, self).__init__()
        self.output_transform = output_transform

    def init(self):
        pass

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        params = self.output_transform(args)
        return self.forward(*params)


class Loss(LossBase):
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(Loss, self).__init__(output_transform)
        self.loss_fn = loss_fn

    def forward(self, *args):
        return self.loss_fn(*args)
