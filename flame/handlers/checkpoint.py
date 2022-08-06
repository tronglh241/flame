import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, MutableMapping, Tuple, Union

import torch
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint as _ModelCheckpoint

from ..engine import Engine
from ..keyword import Keyword
from .handler import Handler


class ModelCheckpoint(Handler):
    def __init__(
        self,
        engine: Engine,
        event: Any,
        modules: List[str],
        dirname: str,
        **kwargs: Any,
    ):
        dirname = os.path.join(dirname, datetime.now().strftime('%y%m%d%H%M%S'))

        self.checkpointer = _ModelCheckpoint(
            dirname,
            **kwargs,
        )

        action = {
            'engine': engine,
            'event': event,
            'func': self.checkpointer,
            'akwargs': {
                'to_save': {module: module for module in modules},
            },
            'eval_kwargs': True,
        }
        super(ModelCheckpoint, self).__init__(actions=[action])

    def state_dict(self) -> MutableMapping:
        return self.checkpointer.state_dict()

    def load_state_dict(self, state_dict: MutableMapping) -> None:
        dirname = Path(self.checkpointer.save_handler.dirname)

        # Fake saved
        for _, file in state_dict['saved']:
            dirname.joinpath(file).touch()

        self.checkpointer.load_state_dict(state_dict)


class BestCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        engine: Engine,
        event: Any,
        modules: List[str],
        dirname: str,
        score_name: str,
        mode: str,
        n_saved: int = 1,
        global_step_transform: Union[Callable, Tuple[Engine, Any]] = None,
        **kwargs: Any,
    ):
        if mode not in {'min', 'max'}:
            raise ValueError(f'Unsupported mode {mode}. Use `min` or `max`.')

        if mode == 'min':
            score_function = lambda engine: - engine.state.metrics[score_name]  # noqa: E731
        else:
            score_function = lambda engine: engine.state.metrics[score_name]  # noqa: E731

        super(BestCheckpoint, self).__init__(
            engine=engine,
            event=event,
            modules=modules,
            dirname=dirname,
            filename_prefix='best',
            score_function=score_function,
            score_name=score_name,
            n_saved=n_saved,
            global_step_transform=global_step_transform,
            include_self=False,
            **kwargs,
        )


class BackupCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        engine: Engine,
        event: Any,
        modules: List[str],
        dirname: str,
        n_saved: int = 1,
        global_step_transform: Union[Callable, Tuple[Engine, Any]] = None,
        **kwargs: Any,
    ):
        super(BackupCheckpoint, self).__init__(
            engine=engine,
            event=event,
            modules=modules,
            dirname=dirname,
            filename_prefix='backup',
            n_saved=n_saved,
            global_step_transform=global_step_transform,
            include_self=True,
            **kwargs,
        )


class CheckpointLoader(Handler):
    def __init__(
        self,
        path: str = None,
        model_key: str = 'model',
        backup_checkpoint_key: str = 'checkpoint.backup',
        **kwargs: Any,
    ):
        actions = []

        if path:
            kwargs['map_location'] = kwargs.pop('map_location', 'cpu')
            checkpoint = torch.load(
                path,
                **kwargs,
            )

            if isinstance(checkpoint, OrderedDict):
                modules = [model_key]
            else:
                if Keyword.CHECKPOINTER in checkpoint:
                    checkpoint[backup_checkpoint_key] = checkpoint.pop(Keyword.CHECKPOINTER)

                modules = list(checkpoint.keys())

            action = {
                'event': Events.STARTED,
                'func': self.load,
                'akwargs': {
                    'modules': modules,
                    'path': path,
                },
                'eval_kwargs': [
                    True,
                    False,
                ],
            }
            actions.append(action)

            self.load_kwargs = kwargs

        super(CheckpointLoader, self).__init__(actions=actions)

    def load(self, modules: list, path: str) -> None:
        checkpoint = torch.load(
            path,
            **self.load_kwargs,
        )

        if isinstance(checkpoint, OrderedDict):
            state_dicts = [checkpoint]
        else:
            state_dicts = checkpoint.values()

        for module, state_dict in zip(modules, state_dicts):
            module.load_state_dict(state_dict)


class ConfigBackup(Handler):
    def __init__(
        self,
        backup_checkpoint: BackupCheckpoint,
    ):
        self.dirname = Path(backup_checkpoint.checkpointer.save_handler.dirname)
        action = {
            'event': Events.STARTED,
            'func': self,
            'akwargs': {
                'config': Keyword.CONFIG,
            },
            'eval_kwargs': True,
        }
        super(ConfigBackup, self).__init__(actions=[action])

    def __call__(
        self,
        config: str,
    ) -> None:
        cfg_file = Path(config)

        with cfg_file.open(mode='r') as fr:
            with self.dirname.joinpath(cfg_file.name).open(mode='w') as fw:
                fw.write(fr.read())
