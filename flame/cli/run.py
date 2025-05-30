from collections import ChainMap
from importlib import import_module
from typing import Callable, List, MutableMapping, Optional, Tuple

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
        'rank': config.get(Keyword.RANK, 0),
        'barrier': config.get(Keyword.BARRIER, False),
    }

    if Keyword.ENGINE in config:
        action_kwargs['engine'] = config[Keyword.ENGINE]

    return Action(**action_kwargs)


def reval(config: CfgNode, context: MutableMapping = None) -> Tuple[CfgNode, List[Action]]:
    '''
    Recursively evaluates a configuration tree, instantiating Python objects and collecting executable actions.

    This function supports structured YAML/Dict configurations that describe objects and handlers
    to be instantiated and evaluated dynamically. It returns the modified configuration object and
    a list of `Action` instances that can be later executed or scheduled in an engine or loop.

    Args:
        config (CfgNode): A hierarchical configuration node (e.g., from a parsed YAML or a nested dict).
        context (MutableMapping, optional): Optional evaluation context used when evaluating string expressions.
            If not provided, `eval()` is performed in a restricted local context.

    Returns:
        Tuple[CfgNode, List[Action]]:
            - The fully evaluated configuration, where any handler/module instantiations or string expressions
              are replaced by actual Python objects.
            - A list of `Action` objects extracted from handlers and explicitly defined config entries.

    Behavior:
        - Dicts: Recursively evaluates values unless keys are in `Keyword.NOT_EVAL`.
        - Lists: Evaluates each element.
        - Strings: Evaluated via `eval()` using the provided context.
        - Module + Name: Imports and instantiates the specified object using optional kwargs.
            May attach associated handler actions and events.
        - Handler + Event: Builds an `Action` from the given callable and event info.

    Raises:
        KeyError: If both (`module`, `name`) and `handler` are provided in the same config block,
            which is considered ambiguous.

    Example:
        ```yaml
        my_handler:
            module: mylib.handlers
            name: MyHandler
            kwargs:
                value: 10
            event: EPOCH_COMPLETED
        ```

        This block will be evaluated into an instance of `mylib.handlers.MyHandler(value=10)`
        and an `Action` scheduled on the `EPOCH_COMPLETED` event.

    Note:
        - Supports integration with custom `Handler` classes that expose `actions` attributes.
        - This evaluation logic enables declarative experiment setup via config files.
    '''
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
        config = eval(config, {**context} if context is not None else None)

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


def simple_run(config: CfgNode) -> MutableMapping:
    engine, actions, context = setup(config)

    for action in actions:
        action.attach(context)

    engine.run()

    return context


def parallel_run(local_rank: int, config: CfgNode) -> MutableMapping:
    return simple_run(config)


def run(file: str) -> Optional[MutableMapping]:
    with open(file) as f:
        config = CfgNode.load_cfg(f)

    config[Keyword.CONFIG] = file
    context = None

    if 'launcher' in config:
        launcher, _ = reval(config.pop('launcher'))

        with launcher:
            launcher.run(parallel_run, config=config)
    else:
        context = simple_run(config)

    return context
