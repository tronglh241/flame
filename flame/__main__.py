import argparse
from typing import Any

from .core.config.config import global_cfg
from .core.engine.engine import Engine
from .handlers.handler_wrapper import HandlerWrapper
from .keywords import ENGINE_KEY, EVENT_KEY, HANDLER_KEY, HANDLER_KWARGS_KEY

SETUP_KEYWORDS = [
    ENGINE_KEY,
    HANDLER_KEY,
    EVENT_KEY,
]


def setup(settings: Any, context: dict) -> None:
    if isinstance(settings, dict):
        if all([key in settings for key in SETUP_KEYWORDS]):
            engine = settings.get(ENGINE_KEY)
            handler = settings.get(HANDLER_KEY)
            event = settings.get(EVENT_KEY)
            kwargs = settings.get(HANDLER_KWARGS_KEY, {})

            if not isinstance(engine, Engine):
                raise TypeError(f'{ENGINE_KEY} needs to be an Engine, {type(engine)} found.')

            if not callable(handler):
                raise TypeError(f'{HANDLER_KEY} needs to be a Callable, {type(handler)} found.')

            engine.add_event_handler(event, HandlerWrapper(handler, context), **kwargs)
        else:
            for value in settings.values():
                setup(value, context)
    elif isinstance(settings, (list, tuple)):
        for value in settings:
            setup(value, context)
    else:
        raise TypeError(f'Only dict, list and tuple are supported in setup procedure, {type(settings)} found.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--engine', '-e', default='core.engine')
    parser.add_argument('--setup', '-s', default='setup')
    args = parser.parse_args()

    config = global_cfg
    config.merge_from_file(args.config_file)

    modules = config.eval()
    engine = modules.get(args.engine)
    settings = modules.get(args.setup)

    if engine is None:
        raise RuntimeError('`engine` cannot be found.')

    if not isinstance(engine, Engine):
        raise TypeError('`engine` must be an instance of core Engine.')

    setup(settings, modules)
    engine.run()
