import argparse

from .core.config.config import global_cfg
from .core.engine.engine import Engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()

    config = global_cfg
    config.merge_from_file(args.config_file)

    modules = config.eval()
    engine = modules.get('engine', None)

    if engine is None:
        raise RuntimeError('`engine` can not be found.')

    if not isinstance(engine, Engine):
        raise TypeError('`engine` must be an instance of core Engine.')

    engine.run()
