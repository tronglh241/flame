import sys
from abc import ABC, abstractmethod, abstractproperty
from typing import MutableMapping

from flame.cli import cli


class Checker(ABC):
    @abstractproperty
    def command(self) -> str:  # pragma: no cover
        pass

    def execute_command(self) -> MutableMapping:
        sys_argv = sys.argv
        sys.argv = self.command.split()
        context = cli(return_context=True)
        sys.argv = sys_argv
        return context

    @abstractmethod
    def check(self) -> None:  # pragma: no cover
        pass
