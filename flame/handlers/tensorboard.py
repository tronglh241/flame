import os
from datetime import datetime
from typing import Any, Dict, List

from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events

from .handler import Handler


class Tensorboard(Handler):
    '''
    A handler to integrate Ignite engines with TensorBoard logging using Ignite's TensorboardLogger.

    This class initializes a TensorBoard logger with an optional log directory timestamped by current datetime,
    attaches multiple logger handlers to specified engines, and ensures proper flushing and closing of
    the TensorBoard writer at appropriate Ignite events.

    Args:
        log_dir (str, optional): Base directory path for TensorBoard logs. If provided, a timestamp
            in the format YYMMDDHHMMSS will be appended to create a unique subdirectory.
        logger_handlers (List[Dict[str, Any]]): A list of dictionaries specifying logger handler configurations.
            Each dictionary can contain any keyword arguments accepted by `TensorboardLogger.attach()`,
            including an optional 'engine' key to specify which Ignite Engine to attach to.
        **kwargs: Additional keyword arguments passed to the `TensorboardLogger` constructor.

    Behavior:
        - Automatically appends a datetime-based suffix to `log_dir` if provided, to create unique runs.
        - Attaches specified logger handlers to their engines at handler initialization.
        - Flushes the TensorBoard writer after every epoch completion.
        - Closes the TensorBoard writer upon completion of training.
    '''

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
