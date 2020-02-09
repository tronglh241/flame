import sys

from . import utils
from importlib import import_module


class Frame(dict):
    def __init__(self, config_path):
        super(Frame, self).__init__()
        self.config_path = config_path


if __name__ == '__main__':
    config_path = sys.argv[1]
    configs = utils.load_yaml(config_path)
    frame = Frame(config_path)

    __extralibs__ = {name: import_module(lib) for (name, lib) in configs.get('extralibs', {}).items()}
    del configs['extralibs']

    for module_name, module_config in configs.items():
        module = utils.create_instance(module_config)
        module.attach(frame, module_name)

    for module in frame.values():
        module.init()

    assert 'engine' in frame, 'The frame does not have engine.'
    frame['engine'].run()