from pathlib import Path

import pytest

from ..checker import Checker


class BasicTrainChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'basic_train.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        self.execute_command()


class BasicTrainNoEngineChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'basic_train_no_engine.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        with pytest.raises(
            KeyError,
            match='No engine found. You must specify `engine` in the config.',
        ):
            self.execute_command()


class BasicTrainWithActionChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'basic_train_with_action.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        self.execute_command()


class BasicTrainWithActionKeyErrorChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'basic_train_with_action_keyerror.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        with pytest.raises(
            KeyError,
            match=r'[If .* and .* are specified, then .* is disallowed, and vice versa.]'
        ):
            self.execute_command()
