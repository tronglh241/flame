import os
from datetime import datetime
from typing import Any, Dict, List

from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events

from .handler import Handler


class Tensorboard(Handler):
    def __init__(
        self,
        *,
        log_dir: str = None,
        logger_handlers: List[Dict[str, Any]],
        **kwargs: Any,
    ):
        if log_dir is not None:
            log_dir = os.path.join(log_dir, datetime.now().strftime('%y%m%d%H%M%S'))

        self.logger = TensorboardLogger(log_dir=log_dir, **kwargs)

        actions = []

        for logger_handler in logger_handlers:
            action = {}

            if 'engine' in logger_handler:
                action.update({
                    'engine': logger_handler.pop('engine'),
                })

            action.update({
                'event': None,
                'func': self.logger.attach,
                'akwargs': logger_handler,
            })
            actions.append(action)

        actions.append({
            'event': Events.EPOCH_COMPLETED,
            'func': self.logger.writer.flush,
        })

        actions.append({
            'event': Events.COMPLETED,
            'func': self.logger.writer.close,
        })

        super(Tensorboard, self).__init__(actions=actions)
