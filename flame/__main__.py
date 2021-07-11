import argparse

from .core.config.config import CfgNode
from .module import Module


class Frame(dict):
    def __init__(self, config_path):
        super(Frame, self).__init__()
        self.config_path = config_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = CfgNode.load_cfg(f)

    frame = Frame(args.config_file)
    modules = config.eval()

    for name, module in modules.items():
        if isinstance(module, Module):
            module.attach(frame, name)
        else:
            frame[name] = module

    for module in frame.values():
        if isinstance(module, Module):
            module.init()

    assert 'engine' in frame, 'The frame does not have engine.'
    frame['engine'].run()
