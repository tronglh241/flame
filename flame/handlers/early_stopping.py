from ignite.handlers import EarlyStopping as _EarlyStopping

from flame.engine import Engine


class EarlyStopping(_EarlyStopping):
    '''
    Custom EarlyStopping handler that monitors a specific metric from an evaluator engine.

    This class extends Ignite's `EarlyStopping` handler by linking the evaluation engine
    explicitly and allowing configuration via a `score_name` and `mode` (`min` or `max`).

    Args:
        patience (int): Number of events to wait if no improvement and then stop the training.
        score_name (str): Name of the metric in `engine.state.metrics` to monitor.
        mode (str): One of {'min', 'max'}. In 'min' mode, early stopping is triggered when
            the monitored metric stops decreasing. In 'max' mode, when it stops increasing.
        trainer (Engine): The training engine to stop when early stopping is triggered.
        evaluator (Engine): The evaluation engine from which metrics are read.
        min_delta (float, optional): Minimum change in the monitored score to qualify as an improvement.
        cumulative_delta (bool, optional): Whether `min_delta` should be accumulated over patience.

    Notes: The evaluator must have computed the desired metric (`score_name`) before this handler is called.
    '''

    def __init__(
        self,
        patience: int,
        score_name: str,
        mode: str,
        trainer: Engine,
        evaluator: Engine,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        if mode not in {'min', 'max'}:
            raise ValueError(f'Unsupported mode {mode}. Use `min` or `max`.')

        if mode == 'min':
            score_function = lambda engine: - engine.state.metrics[score_name]  # noqa: E731
        else:
            score_function = lambda engine: engine.state.metrics[score_name]  # noqa: E731

        self.evaluator = evaluator

        super(EarlyStopping, self).__init__(
            patience=patience,
            score_function=score_function,
            trainer=trainer,
            min_delta=min_delta,
            cumulative_delta=cumulative_delta,
        )

    def __call__(self) -> None:
        super(EarlyStopping, self).__call__(self.evaluator)
