from collections import ChainMap
from importlib import import_module
from typing import Callable, List, MutableMapping, Tuple

from yacs.config import CfgNode

from ..engine import Engine
from ..handlers import Action, Handler
from ..keyword import Keyword


def create_action(func: Callable, config: CfgNode) -> Action:
    action_kwargs = {
        'event': config[Keyword.EVENT],
        'func': func,
        'akwargs': config.get(Keyword.AKWARGS, {}),
        'eval_kwargs': True,
    }

    if Keyword.ENGINE in config:
        action_kwargs['engine'] = config[Keyword.ENGINE]

    return Action(**action_kwargs)


def reval(config: CfgNode, context: MutableMapping = None) -> Tuple[CfgNode, List[Action]]:
    actions = []

    if isinstance(config, dict):
        for key, value in config.items():
            if key not in Keyword.NOT_EVAL.values():
                config[key], sub_actions = reval(value, context)
                actions.extend(sub_actions)

        if Keyword.MODULE in config and Keyword.NAME in config and Keyword.HANDLER in config:
            raise KeyError(f'If {Keyword.MODULE} and {Keyword.NAME} are specified,'
                           f'then {Keyword.HANDLER} is disallowed, and vice versa.')

        if Keyword.MODULE in config and Keyword.NAME in config:
            module = config.pop(Keyword.MODULE)
            name = config.pop(Keyword.NAME)
            config_kwargs = config.pop(Keyword.KWARGS, {})
            obj = eval(name, {}, vars(import_module(module)))(**config_kwargs)

            if Keyword.EVENT in config:
                func = getattr(obj, config[Keyword.FUNCTION]) if Keyword.FUNCTION in config else obj
                actions.append(create_action(func, config))

            if isinstance(obj, Handler):
                actions.extend(obj.actions)

            config = obj

        elif Keyword.HANDLER in config and Keyword.EVENT in config:
            actions.append(create_action(config[Keyword.HANDLER], config))

    elif isinstance(config, list):
        eles: list = []

        for ele in config:
            ele, sub_actions = reval(ele, context)
            eles.append(ele)
            actions.extend(sub_actions)

        config = eles

    elif isinstance(config, str):
        config = eval(config, {}, context)

    return config, actions


def setup(config: CfgNode) -> Tuple[Engine, List[Action], MutableMapping]:
    config = config.clone()
    extralibs = {}

    # Generate extra libs
    for alias, lib_info in config.pop(Keyword.EXTRALIBS, {}).items():
        if isinstance(lib_info, dict):
            module = lib_info[Keyword.MODULE]
            name = lib_info[Keyword.NAME]
            lib = getattr(import_module(module), name)
        else:
            lib = import_module(lib_info)

        extralibs[alias] = lib

    # Eval config
    context = ChainMap(config, extralibs)
    config, actions = reval(config, context)

    if extralibs:
        config[Keyword.EXTRALIBS] = extralibs

    engine = config.get('engine')

    if not isinstance(engine, Engine):
        raise KeyError('No engine found. You must specify `engine` in the config.')

    return engine, actions, context


def run(file: str) -> MutableMapping:
    with open(file) as f:
        config = CfgNode.load_cfg(f)

    config[Keyword.CONFIG] = file
    engine, actions, context = setup(config)

    for action in actions:
        action.attach(context)

    engine.run()
    return context
