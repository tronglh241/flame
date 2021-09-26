from ignite.handlers import EarlyStopping as _EarlyStopping

from flame.core.engine.engine import Engine


class EarlyStopping(_EarlyStopping):
    def __init__(self, patience: int, score_name: str, mode: str, trainer: Engine, min_delta: float = 0.0,
                 cumulative_delta: bool = False):
        if mode not in {'min', 'max'}:
            raise ValueError(f'mode {mode} is unknown!')

        if mode == 'min':
            score_function = lambda engine: - engine.state.metrics[score_name]  # noqa: E731
        else:
            score_function = lambda engine: engine.state.metrics[score_name]  # noqa: E731

        super(EarlyStopping, self).__init__(patience, score_function, trainer, min_delta, cumulative_delta)
