import time
from typing import Any, Dict, List, Union

from ignite.contrib.handlers import ProgressBar as _ProgressBar
from ignite.engine import Events
from tqdm import tqdm

from ..engine import Engine
from .handler import Handler


class ProgressBar(Handler):
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
        msg = f'Epoch #{engine.state.epoch} - {time.asctime()} - '

        for evaluator_name, evaluator in evaluators.items():
            for metric_name, metric_value in evaluator.state.metrics.items():
                if self.metric_names and metric_name in self.metric_names:
                    msg += f'{evaluator_name}_{metric_name}: {metric_value:.4f} - '

        tqdm.write(msg[:-3])
