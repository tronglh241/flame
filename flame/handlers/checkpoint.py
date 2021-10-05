import os
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Mapping, MutableMapping

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint as _ModelCheckpoint
from ignite.handlers import global_step_from_engine

from flame.core.config.config import global_cfg
from flame.core.engine.engine import Engine
from flame.keywords import CONFIG_KEY, DEFAULT_ENGINE, DEFAULT_MODEL


class ModelCheckpoint(_ModelCheckpoint):
    def __init__(self, dirname: str, saved_modules: Mapping, global_step_transform: Callable = None, **kwargs):
        dirname = os.path.join(dirname, datetime.now().strftime('%y%m%d%H%M'))

        if global_step_transform is None:
            engine = global_cfg.eval_key(DEFAULT_ENGINE)
            global_step_transform = global_step_from_engine(engine, Events.EPOCH_STARTED)

        self.saved_modules = saved_modules
        super(ModelCheckpoint, self).__init__(dirname, global_step_transform=global_step_transform, **kwargs)

    def load_state_dict(self, state_dict: MutableMapping) -> None:
        dirname = Path(self.save_handler.dirname)

        # Fake saved
        for _, file in state_dict['saved']:
            dirname.joinpath(file).touch()

        super(ModelCheckpoint, self).load_state_dict(state_dict)

    def __call__(self, engine: Engine) -> None:
        super(ModelCheckpoint, self).__call__(engine, self.saved_modules)


class BestCheckpointer(ModelCheckpoint):
    def __init__(self, dirname: str, score_name: str, mode: str, n_saved: int = 1, saved_module_keys: str = None,
                 global_step_transform: Callable = None, **kwargs):
        if mode not in {'min', 'max'}:
            raise ValueError(f'mode {mode} is unknown!')

        if mode == 'min':
            score_function = lambda engine: - engine.state.metrics[score_name]  # noqa: E731
        else:
            score_function = lambda engine: engine.state.metrics[score_name]  # noqa: E731

        if saved_module_keys is None:
            saved_modules = {
                'model': global_cfg.eval_key(DEFAULT_MODEL)
            }
        else:
            saved_modules = {k: global_cfg.eval_key(k) for k in saved_module_keys}

        super(BestCheckpointer, self).__init__(dirname, saved_modules, filename_prefix='best',
                                               score_function=score_function, score_name=score_name, n_saved=n_saved,
                                               global_step_transform=global_step_transform, include_self=False,
                                               **kwargs)


class BackupCheckpointer(ModelCheckpoint):
    def __init__(self, dirname: str, saved_module_keys: List[str], n_saved: int = 1,
                 global_step_transform: Callable = None, **kwargs):
        saved_modules = {k: global_cfg.eval_key(k) for k in saved_module_keys}
        saved_modules[CONFIG_KEY] = global_cfg

        super(BackupCheckpointer, self).__init__(dirname, saved_modules, filename_prefix='backup', n_saved=n_saved,
                                                 global_step_transform=global_step_transform, include_self=True,
                                                 **kwargs)
