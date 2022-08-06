from ignite.handlers import EarlyStopping as _EarlyStopping

from flame.engine import Engine


class EarlyStopping(_EarlyStopping):
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
