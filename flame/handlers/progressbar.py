import time
from typing import Any, Dict, List, Union

from ignite.contrib.handlers import ProgressBar as _ProgressBar
from ignite.engine import Events
from tqdm import tqdm

from ..engine import Engine
from .handler import Handler


class ProgressBar(Handler):
    '''
    A handler that attaches progress bars to a trainer and multiple evaluator engines using Ignite's
    contrib ProgressBar, and optionally logs evaluator metrics after each epoch.

    This handler helps visualize training progress for the trainer engine and evaluation progress for
    one or more evaluator engines. It supports configuring progress bar parameters separately for the
    trainer and evaluators. Additionally, it can print evaluator metrics to the console at the end of
    each evaluation epoch using `tqdm.write`.

    Args:
        trainer (Engine, optional): Ignite Engine instance for training. If provided, a progress bar
            is attached to this engine.
        evaluators (Dict[str, Engine], optional): A dictionary mapping evaluator names to Ignite Engine
            instances. Progress bars will be attached to all evaluators.
        trainer_pbar_kwargs (Dict[str, Any], optional): Keyword arguments passed to the trainer's
            progress bar constructor. Defaults to an empty dict.
        evaluators_pbar_kwargs (Union[Dict[str, Any], List[Dict[str, Any]]], optional): Keyword arguments
            for evaluators' progress bars. Can be a single dict applied to all evaluators or a list of dicts
            specifying each evaluator's arguments. Defaults to `{'desc': 'Evaluating'}`.
        trainer_pbar_akwargs (Dict[str, Any], optional): Additional keyword arguments passed to the
            `attach` method of the trainer progress bar, such as `output_transform`.
        evaluators_pbar_akwargs (Union[Dict[str, Any], List[Dict[str, Any]]], optional): Additional keyword
            arguments for evaluators' progress bars passed to their `attach` methods.
        metric_names (List[str], optional): List of metric names to log after evaluation epochs.
            If `None`, logs all metrics.

    Example:
        ```python
        progress_bar = ProgressBar(
            trainer=trainer_engine,
            evaluators={'val': val_engine, 'test': test_engine},
            metric_names=['accuracy', 'loss']
        )
        ```

    Notes:
        - Logs evaluator metrics at the end of each evaluation epoch using `tqdm.write`.
        - Supports attaching progress bars to multiple engines in distributed or single-process settings.
        - Uses Ignite's `ProgressBar` from `ignite.contrib.handlers`.
    '''

    def __init__(
        self,
        trainer: Engine = None,
        evaluators: Dict[str, Engine] = None,
        trainer_pbar_kwargs: Dict[str, Any] = None,
        evaluators_pbar_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
        trainer_pbar_akwargs: Dict[str, Any] = None,
        evaluators_pbar_akwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
        metric_names: List[str] = None,
    ):
        self.metric_names = metric_names

        default_trainer_pbar_kwargs: Dict[str, Any] = {
        }
        default_evaluator_pbar_kwargs: Dict[str, Any] = {
            'desc': 'Evaluating',
        }
        default_trainer_pbar_akwargs: Dict[str, Any] = {
            'output_transform': lambda x: {'loss': x},
        }
        default_evaluator_pbar_akwargs: Dict[str, Any] = {
        }

        actions: List[Dict[str, Any]] = []
        engines: List[Engine] = []
        pbars_kwargs = []
        pbars_akwargs = []

        if trainer:
            engines.append(trainer)

            if trainer_pbar_kwargs is None:
                pbars_kwargs.append(default_trainer_pbar_kwargs)
            else:
                pbars_kwargs.append({**default_trainer_pbar_kwargs, **trainer_pbar_kwargs})

            if trainer_pbar_akwargs is None:
                pbars_akwargs.append(default_trainer_pbar_akwargs)
            else:
                pbars_akwargs.append({**default_trainer_pbar_akwargs, **trainer_pbar_akwargs})

        if evaluators:
            engines.extend(evaluators.values())

            if evaluators_pbar_kwargs is None:
                pbars_kwargs.extend([default_evaluator_pbar_kwargs for _ in evaluators])
            else:
                if not isinstance(evaluators_pbar_kwargs, list):
                    evaluators_pbar_kwargs = [evaluators_pbar_kwargs for _ in evaluators]

                pbars_kwargs.extend({**default_evaluator_pbar_kwargs, **evaluator_pbar_kwargs}
                                    for evaluator_pbar_kwargs in evaluators_pbar_kwargs)

            if evaluators_pbar_akwargs is None:
                pbars_akwargs.extend([default_evaluator_pbar_akwargs for _ in evaluators])
            else:
                if not isinstance(evaluators_pbar_akwargs, list):
                    evaluators_pbar_akwargs = [evaluators_pbar_akwargs for _ in evaluators]

                pbars_akwargs.extend({**default_evaluator_pbar_akwargs, **evaluator_pbar_akwargs}
                                     for evaluator_pbar_akwargs in evaluators_pbar_akwargs)

        for engine, pbar_kwargs, pbar_akwargs in zip(engines, pbars_kwargs, pbars_akwargs):
            pbar = _ProgressBar(**pbar_kwargs)
            actions.append({
                'event': None,
                'func': pbar.attach,
                'akwargs': {
                    'engine': engine,
                    **pbar_akwargs,
                },
            })

        if evaluators:
            actions.append({
                'event': Events.EPOCH_COMPLETED,
                'func': self.log_metrics,
                'akwargs': {
                    'evaluators': evaluators,
                }
            })

        super(ProgressBar, self).__init__(actions=actions)

    def log_metrics(self, engine: Engine, evaluators: Dict[str, Engine]) -> None:
        msg = f'Epoch #{engine.state.epoch} - {time.asctime()}'
        metric_msgs = ['']

        for evaluator_name, evaluator in evaluators.items():
            for metric_name, metric_value in evaluator.state.metrics.items():
                if self.metric_names is None or metric_name in self.metric_names:
                    if isinstance(metric_value, dict):
                        for name, value in metric_value.items():
                            metric_msgs.append(f'{evaluator_name}_{metric_name}_{name}: {value:.4f}')
                    else:
                        metric_msgs.append(f'{evaluator_name}_{metric_name}: {metric_value:.4f}')

        tqdm.write(msg + ' - '.join(metric_msgs))
