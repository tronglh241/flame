import os
import shutil
from pathlib import Path

import pytest

from .checker import Checker


class InitCwdChecker(Checker):
    command = 'flame init'

    def check(self):
        self.execute_command()
        cwd = Path(os.getcwd())

        assert cwd.joinpath('configs', 'train.yml').exists()
        assert cwd.joinpath('configs', 'test.yml').exists()

        shutil.rmtree(cwd.joinpath('configs'))


class InitCwdFullChecker(Checker):
    command = 'flame init -f'

    def check(self):
        self.execute_command()
        cwd = Path(os.getcwd())

        assert cwd.joinpath('configs', 'train.yml').exists()
        assert cwd.joinpath('configs', 'test.yml').exists()
        assert cwd.joinpath('run.py').exists()

        shutil.rmtree(cwd.joinpath('configs'))
        cwd.joinpath('run.py').unlink()


class InitChecker(Checker):
    command = 'flame init test_repo'

    def check(self):
        self.execute_command()
        dirname = Path('test_repo')

        assert dirname.joinpath('configs', 'train.yml').exists()
        assert dirname.joinpath('configs', 'test.yml').exists()

        shutil.rmtree(dirname)


class InitFullChecker(Checker):
    command = 'flame init -f test_repo'

    def check(self):
        self.execute_command()
        dirname = Path('test_repo')

        assert dirname.joinpath('configs', 'train.yml').exists()
        assert dirname.joinpath('configs', 'test.yml').exists()
        assert dirname.joinpath('run.py').exists()

        shutil.rmtree(dirname)


class InitExistedChecker(Checker):
    command = 'flame init -f test_repo'

    def check(self):
        dirname = Path('test_repo')
        dirname.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileExistsError):
            self.execute_command()

        shutil.rmtree(dirname)
