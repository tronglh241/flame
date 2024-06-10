from pathlib import Path

from ...checker import Checker


class EarlyStoppingMaxChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'early_stopping_max.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        context = self.execute_command()

        assert context is not None
        engine = context['engine']

        assert engine.state.epoch < engine.state.max_epochs


class EarlyStoppingMinChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'early_stopping_min.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        context = self.execute_command()

        assert context is not None
        engine = context['engine']

        assert engine.state.epoch < engine.state.max_epochs
