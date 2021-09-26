from typing import Dict, List

from ignite.metrics.metric import Metric

from flame.core.engine.engine import Engine


class Metrics:
    def __init__(self, evaluators: List[Engine], metrics: Dict[str, Metric]):
        super(Metrics, self).__init__()
        self.evaluators = evaluators
        self.metrics = metrics

    def __call__(self) -> None:
        for evaluator in self.evaluators:
            for name, metric in self.metrics.items():
                metric.attach(evaluator, name)
