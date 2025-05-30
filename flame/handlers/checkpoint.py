import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, MutableMapping

import torch
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint as _ModelCheckpoint
from torch import nn

from ..engine import Engine
from ..keyword import Keyword
from .handler import Handler


class ModelCheckpoint(Handler):
    '''
    Handler for saving model checkpoints during training or evaluation.

    Wraps around Ignite's ModelCheckpoint handler and integrates with a custom Engine.

    Args:
        engine (Engine): The engine instance to attach the checkpoint handler.
        event (Any): The Ignite event to trigger checkpointing (e.g., Events.EPOCH_COMPLETED).
        modules (List[str]): List of module names (strings) to save checkpoints for.
        dirname (str): Directory path where checkpoints will be saved. A timestamp subdirectory
            (format: 'yymmddHHMMSS') will be appended automatically.
        **kwargs (Any): Additional keyword arguments forwarded to Ignite's ModelCheckpoint.
    '''

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
        for _, file in state_dict['_saved']:
            dirname.joinpath(file).touch()

        self.checkpointer.load_state_dict(state_dict)


class BestCheckpoint(ModelCheckpoint):
    '''
    Handler for saving the best model checkpoints based on a specific metric.

    Extends `ModelCheckpoint` to only save checkpoints when a monitored score improves,
    according to the specified mode ('min' or 'max').

    Args:
        engine (Engine): The Ignite engine to attach the checkpoint handler.
        event (Any): The event that triggers the checkpoint saving (e.g., Events.EPOCH_COMPLETED).
        modules (List[str]): List of module names (strings) to checkpoint.
        dirname (str): Directory to save checkpoints into. A timestamp subdirectory will be created.
        score_name (str): Name of the metric in `engine.state.metrics` to monitor.
        mode (str): One of `'min'` or `'max'`. `'min'` saves when the score decreases,
            `'max'` saves when the score increases.
        n_saved (int, optional): Number of best checkpoints to keep. Defaults to 1.
        global_step_transform (Callable[[Engine, Any], int], optional): A function to extract a global step
            from the engine. This is used in the filename. Defaults to None.
        **kwargs (Any): Additional keyword arguments forwarded to Ignite's `ModelCheckpoint`.
    '''

    def __init__(
        self,
        engine: Engine,
        event: Any,
        modules: List[str],
        dirname: str,
        score_name: str,
        mode: str,
        n_saved: int = 1,
        global_step_transform: Callable[[Engine, Any], int] = None,
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
    '''
    Handler for periodically saving backup checkpoints of models during training.

    This class extends `ModelCheckpoint` and is intended to save model checkpoints
    regularly (e.g., every few epochs or iterations), regardless of performance metrics.

    Args:
        engine (Engine): The Ignite engine to attach the checkpoint handler.
        event (Any): The event that triggers the checkpoint saving (e.g., Events.EPOCH_COMPLETED).
        modules (List[str]): List of module names (strings) to checkpoint.
        dirname (str): Directory to save checkpoints into. A timestamp subdirectory will be created.
        n_saved (int, optional): Number of most recent checkpoints to keep. Defaults to 1.
        global_step_transform (Callable[[Engine, Any], int], optional): A function to extract a global step
            from the engine. This is used in the filename. Defaults to None.
        **kwargs (Any): Additional keyword arguments forwarded to Ignite's `ModelCheckpoint`.
    '''

    def __init__(
        self,
        engine: Engine,
        event: Any,
        modules: List[str],
        dirname: str,
        n_saved: int = 1,
        global_step_transform: Callable[[Engine, Any], int] = None,
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
    '''
    Handler for loading model or checkpoint states from a given path at engine start.

    This class supports loading either a single model's `state_dict` or a full checkpoint
    dictionary. It can be attached to an Ignite engine to automatically restore
    model weights (and optionally other modules) at the beginning of training or evaluation.

    Args:
        path (str, optional): Path to the checkpoint file to load. If not provided, no loading occurs.
        model_key (str, optional): Key in the YAML config that refers to the model module.
            Used when loading a single model's `state_dict`. Defaults to 'model'.
        backup_checkpoint_key (str, optional): Key in the YAML config that refers to the backup checkpoint handler.
            Defaults to 'checkpoint.backup'.
        is_model (bool, optional): If True, the checkpoint is treated as a model `state_dict`. If False,
            it's assumed to be a full checkpoint. If None, it is inferred based on checkpoint structure.
        **kwargs (Any): Additional arguments passed to `torch.load`, e.g., `map_location`.
    '''

    def __init__(
        self,
        path: str = None,
        model_key: str = 'model',
        backup_checkpoint_key: str = 'checkpoint.backup',
        is_model: bool = None,
        **kwargs: Any,
    ):
        actions = []

        if path:
            kwargs['map_location'] = kwargs.pop('map_location', 'cpu')
            checkpoint = torch.load(
                path,
                **kwargs,
            )

            if is_model is None and isinstance(checkpoint, OrderedDict) or is_model:
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
                'rank': -1,
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
            if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                module = module.module

            module.load_state_dict(state_dict)


class ConfigBackup(Handler):
    '''
    Handler for backing up the configuration file alongside model checkpoints.

    This class copies the YAML configuration file used for the experiment into
    the same directory where the backup checkpoints are stored. It is triggered
    at the start of the engine's execution.

    Args:
        backup_checkpoint (BackupCheckpoint): An instance of BackupCheckpoint whose
            directory is used for storing the configuration file.
    '''

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
