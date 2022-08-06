from typing import Dict, List

from ignite.metrics import Metric

from ..engine import Engine
from .handler import Handler


class Metrics(Handler):
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
        }
        super(Metrics, self).__init__(actions=[action])

    def __call__(self) -> None:
        for evaluator in self.evaluators:
            for name, metric in self.metrics.items():
                metric.attach(evaluator, name)
