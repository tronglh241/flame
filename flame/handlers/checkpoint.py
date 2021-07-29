import os
from datetime import datetime
from pathlib import Path
from typing import MutableMapping

from ignite.handlers import ModelCheckpoint as _ModelCheckpoint


class ModelCheckpoint(_ModelCheckpoint):
    def __init__(self, dirname: str, **kwargs):
        dirname = os.path.join(dirname, datetime.now().strftime('%y%m%d%H%M'))
        super(ModelCheckpoint, self).__init__(dirname, **kwargs)

    def load_state_dict(self, state_dict: MutableMapping) -> None:
        dirname = Path(self.save_handler.dirname)

        # Fake saved
        for _, file in state_dict['saved']:
            dirname.joinpath(file).touch()

        super(ModelCheckpoint, self).load_state_dict(state_dict)
