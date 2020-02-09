from .... import utils
from ...module import Module
from importlib import import_module
from ignite.engine import Events


class Metrics(Module):
    def __init__(self, metric_module, metric_class, metric_name, metric_params, attach_to=None):
        super(Metrics, self).__init__()

        if not (len(metric_module) == len(metric_class) and len(metric_module) == len(metric_name) and len(metric_module) == len(metric_params)):
            raise ValueError('Length of metric_module, metric_class, metric_name, metric_params must be the same.')
        
        self.metrics = utils.create_metrics(metric_module, metric_class, metric_name, metric_params)
        self.metric_values = {}
        self.attach_to = attach_to if attach_to else {}

    def init(self):
        assert all(map(lambda x: x in self.frame, self.attach_to.keys())), f'The frame does not have all {self.attach_to.keys()}.'
        for evaluator, eval_name in self.attach_to.items():
            evaluator = self.frame[evaluator]
            evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, self._save_eval_result, eval_name)
            for metric_name, metric in self.metrics.items():
                metric.attach(evaluator.engine, metric_name)

    def _save_eval_result(self, engine, eval_name):
        self.metric_values[eval_name] = engine.state.metrics