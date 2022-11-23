from __future__ import annotations

import argparse
from enum import Enum
from typing import Callable, Dict, MutableMapping, Optional

from .init import init
from .run import run


class Cmd(str, Enum):
    INIT = 'init'
    RUN = 'run'

    @staticmethod
    def cmds() -> Dict[Cmd, Callable]:
        commands: Dict[Cmd, Callable] = {
            Cmd.INIT: init,
            Cmd.RUN: run,
        }
        return commands

    @staticmethod
    def executor(cmd: Cmd) -> Callable:
        return Cmd.cmds()[cmd]


def cli(return_context: bool = False) -> Optional[MutableMapping]:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    init_parser = subparsers.add_parser(Cmd.INIT.value)
    init_parser.add_argument(
        'directory',
        nargs='?',
        help='''Directory in which the new project is initialized.
            If not specified, it will be initialized in the current directory.''',
    )
    init_parser.add_argument(
        '-f', '--full',
        action='store_true',
        help='Whether to create a full template project or not.'
    )

    run_parser = subparsers.add_parser(Cmd.RUN.value)
    run_parser.add_argument(
        'file',
        help='Config file',
    )

    args = vars(parser.parse_args())
    cmd_executor = Cmd.executor(args.pop('cmd'))

    context = cmd_executor(**args)

    if not return_context:
        context = None

    return context
