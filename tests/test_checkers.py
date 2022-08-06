from typing import Type

import pytest
from checkers.basic_train_checker.basic_train_checker import (
    BasicTrainChecker, BasicTrainNoEngineChecker, BasicTrainWithActionChecker,
    BasicTrainWithActionKeyErrorChecker)
from checkers.checker import Checker
from checkers.handlers_checker.checkpoint.checkpoint import (
    BackupCheckpointChecker, BestCheckpointChecker,
    BestCheckpointWrongModeChecker, CheckpointLoaderModelChecker)
from checkers.handlers_checker.early_stopping.early_stopping import (
    EarlyStoppingMaxChecker, EarlyStoppingMinChecker)
from checkers.init_checker import (InitChecker, InitCwdChecker,
                                   InitCwdFullChecker, InitExistedChecker,
                                   InitFullChecker)


@pytest.mark.parametrize(
    'checker',
    [
        InitCwdChecker,
        InitCwdFullChecker,
        InitChecker,
        InitFullChecker,
        InitExistedChecker,
        BasicTrainChecker,
        BasicTrainNoEngineChecker,
        BasicTrainWithActionChecker,
        BasicTrainWithActionKeyErrorChecker,
        BestCheckpointChecker,
        BestCheckpointWrongModeChecker,
        BackupCheckpointChecker,
        CheckpointLoaderModelChecker,
        EarlyStoppingMaxChecker,
        EarlyStoppingMinChecker,
    ],
)
def test_command(checker: Type[Checker]) -> None:
    checker().check()
