from typing import Dict, List

from ignite.metrics import Metric

from ..engine import Engine
from .handler import Handler


class Metrics(Handler):
    '''
    Attaches a collection of Ignite `Metric` instances to one or more evaluator engines.

    This class wraps the process of registering multiple metrics to multiple `ignite.engine.Engine`
    instances (typically evaluators), ensuring this is done at setup time via the unified `Handler` mechanism.

    Args:
        evaluators (List[Engine]): A list of evaluator engines to which the metrics will be attached.
        metrics (Dict[str, Metric]): A dictionary mapping metric names to `ignite.metrics.Metric` instances
            to be attached to each evaluator.

    Example:
        ```python
        accuracy = Accuracy()
        loss = Loss(loss_fn)
        metrics_handler = Metrics(
            evaluators=[evaluator1, evaluator2],
            metrics={'acc': accuracy, 'loss': loss}
        )
        metrics_handler()  # Attaches metrics immediately
        ```

    Notes:
        - Metrics are attached immediately upon invocation (`__call__`), not deferred to a specific engine event.
        - Runs on all distributed ranks (`rank = -1`).
    '''

    def __init__(
        self,
        evaluators: List[Engine],
        metrics: Dict[str, Metric],
    ):
        self.evaluators = evaluators
        self.metrics = metrics
        action = {
            'event': None,
            'func': self,
            'rank': -1,
        }
        super(Metrics, self).__init__(actions=[action])

    def __call__(self) -> None:
        for evaluator in self.evaluators:
            for name, metric in self.metrics.items():
                metric.attach(evaluator, name)
