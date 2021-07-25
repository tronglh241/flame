from typing import Any, Callable, Optional, Tuple

from ignite.engine.utils import _check_signature

from ..core.config.config import CfgNode
from ..core.engine.engine import Engine


class HandlerWrapper:
    def __init__(self, handler: Callable, context: dict):
        super(HandlerWrapper, self).__init__()
        self.handler = handler
        self.context = context
        self._needs_engine: Optional[bool] = None

    def _get_args_kwargs(self, engine_: Engine, **kwargs: Any) -> Tuple[tuple, dict]:
        kwargs = CfgNode._eval(kwargs, {}, self.context, eval_all=True)
        if self._needs_engine is None:
            try:
                _check_signature(self.handler, 'handler', engine_, **kwargs)
                self._needs_engine = True
            except ValueError:
                self._needs_engine = False
        args = (engine_,) if self._needs_engine else ()
        return args, kwargs

    def __call__(self, engine_: Engine, **kwargs: Any) -> None:
        args, kwargs = self._get_args_kwargs(engine_, **kwargs)
        self.handler(*args, **kwargs)
