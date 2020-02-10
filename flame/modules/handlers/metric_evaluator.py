from ..core.engine.engine import Evaluator
from ignite.engine import Events


class MetricEvaluator(Evaluator):
    def __init__(self, dataset_name, device):
        super(MetricEvaluator, self).__init__(dataset_name, device)
    
    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.model = self.frame['model'].to(self.device)
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self._run)

    def _run(self, engine):
        self.run()