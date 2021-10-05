import argparse
from typing import Any, Mapping

import torch

from .core.config.config import CfgNode, global_cfg
from .core.engine.engine import Engine
from .handlers.handler_wrapper import HandlerWrapper
from .keywords import (CHECKPOINTER_KEY, CONFIG_KEY,
                       DEFAULT_BACKUP_CHECKPOINTER, DEFAULT_ENGINE,
                       DEFAULT_MODEL, ENGINE_KEY, EVENT_KEY, HANDLER_KEY,
                       HANDLER_KWARGS_KEY)

SETUP_KEYWORDS = [
    ENGINE_KEY,
    HANDLER_KEY,
    EVENT_KEY,
]


def build_modules(config_path: str = None, checkpoint_path: str = None, model_key: str = DEFAULT_MODEL,
                  checkpointer_key: str = DEFAULT_BACKUP_CHECKPOINTER, strict: bool = True) -> Any:
    def _load_module(key: str, state_dict: Mapping):
        module = global_cfg.eval_key(key)

        if module is not None:
            module.load_state_dict(state_dict)
        elif strict:
            raise NameError(f'{key} cannot be found.')

    if config_path is None and checkpoint_path is None:
        raise RuntimeError('`config_path` or `checkpoint_path` must be specified.')

    if config_path is not None:
        global_cfg.merge_from_file(config_path)
        modules, extralibs = global_cfg.eval()

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)

            try:
                _load_module(model_key, checkpoint)
            except Exception:
                for key, state_dict in checkpoint.items():
                    _load_module(key, state_dict)
    else:
        checkpoint = torch.load(checkpoint_path)
        global_cfg.merge_from_other_cfg(CfgNode.load_cfg(checkpoint.pop(CONFIG_KEY)))
        modules, extralibs = global_cfg.eval()
        checkpointer = checkpoint.pop(CHECKPOINTER_KEY, None)

        if checkpointer is not None:
            checkpoint[checkpointer_key] = checkpointer

        for key, state_dict in checkpoint.items():
            _load_module(key, state_dict)

    return modules, extralibs


def setup(settings: Any, global_context: dict, local_context: dict) -> None:
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

            engine.add_event_handler(event, HandlerWrapper(handler, global_context, local_context), **kwargs)
        else:
            for value in settings.values():
                setup(value, global_context, local_context)
    elif isinstance(settings, (list, tuple)):
        for value in settings:
            setup(value, global_context, local_context)
    else:
        raise TypeError(f'Only dict, list and tuple are supported in setup procedure, {type(settings)} found.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', default=None)
    parser.add_argument('--checkpoint', '-p', default=None)
    parser.add_argument('--engine', default=DEFAULT_ENGINE)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--checkpointer', default=DEFAULT_BACKUP_CHECKPOINTER)
    parser.add_argument('--not-strict', action='store_true')
    parser.add_argument('--setup', default='setup')
    args = parser.parse_args()

    modules, extralibs = build_modules(args.config, args.checkpoint, args.model, args.checkpointer, not args.not_strict)
    engine = global_cfg.eval_key(args.engine)
    settings = global_cfg.eval_key(args.setup)

    if engine is None:
        raise RuntimeError('`engine` cannot be found.')

    if not isinstance(engine, Engine):
        raise TypeError('`engine` must be an instance of core Engine.')

    setup(settings, extralibs, modules)
    engine.run()
