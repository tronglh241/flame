from typing import Any, Callable, Dict, List, MutableMapping, Union

import ignite.distributed as idist
from ignite.engine.utils import _check_signature

from ..engine import Engine


def reval(
    node: Any,
    context: MutableMapping = None,
) -> Any:
    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = reval(value, context)

    elif isinstance(node, list):
        node = [reval(ele, context) for ele in node]

    elif isinstance(node, str):
        node = eval(node, {}, context)

    return node


class Action:
    def __init__(
        self,
        *,
        engine: Union[str, Engine] = 'engine',
        event: Any,
        func: Callable,
        akwargs: Dict[str, Any] = None,
        eval_kwargs: Union[bool, List[bool]] = False,
        env: MutableMapping = None,
        rank: int = 0,
        barrier: bool = False,
    ):
        super(Action, self).__init__()
        self.event = event
        self.engine = engine
        self.func = func
        self.kwargs = akwargs if akwargs is not None else {}
        self.eval_kwargs = eval_kwargs
        self.env = env
        self.rank = rank
        self.barrier = barrier
        self.needs_engine = True

    def attach(self, env: MutableMapping = None) -> None:
        env = {} if env is None else env
        env = env if self.env is None else env.update(self.env)
        self.env = env if env else self.env

        if isinstance(self.engine, str):
            self.engine = eval(self.engine, {}, self.env)

        if not isinstance(self.engine, Engine):
            raise ValueError(f'`engine` {self.engine} must be an Engine or a key specifying an Engine in config.')

        self.needs_engine = self.check_signature()

        if self.rank >= 0:
            self.func = idist.one_rank_only(
                rank=self.rank,
                with_barrier=self.barrier,
            )(self.func)

        if self.event is not None:
            if not self.engine.has_event_handler(self, self.event):
                self.engine.add_event_handler(self.event, self, **self.kwargs)
        else:
            self(**self.kwargs)

    def check_signature(self) -> bool:
        if not callable(self.func):
            raise TypeError(f'`func` {self.func} is not callable.')

        try:
            _check_signature(self.func, 'func', self.engine, **self.kwargs)
            return True
        except ValueError:
            return False

        raise Exception('Cannot determine whether `func` needs `engine` or not.')

    def __call__(self, **kwargs: Any) -> None:
        if self.eval_kwargs:
            if isinstance(self.eval_kwargs, list):
                kwargs = {k: reval(v, self.env) if self.eval_kwargs[i] else v
                          for i, (k, v) in enumerate(kwargs.items())}
            else:
                kwargs = reval(kwargs, self.env)

        if self.needs_engine:
            kwargs['engine'] = self.engine

        self.func(**kwargs)


class Handler:
    def __init__(
        self,
        actions: List[Dict[str, Any]],
    ):
        self.actions = [Action(**action) for action in actions]
