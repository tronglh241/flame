import os
from pathlib import Path
from typing import Optional

TEMPLATE_DIR = 'template'
TEMPLATE_SUFFIX = '-tpl'
IGNORED_FILES = [
    'run.py-tpl',
]


def init(directory: Optional[str], full: bool) -> None:
    template_dir = Path(__file__).parent.with_name(TEMPLATE_DIR)

    if directory:
        target_dir = Path(directory)

        if target_dir.exists():
            raise FileExistsError(f'{target_dir} already exists.')

        target_dir.mkdir(parents=True)
    else:
        target_dir = Path(os.getcwd())

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(template_dir)):
        _dir = target_dir.joinpath(Path(dirpath).relative_to(template_dir))

        if i:
            _dir.mkdir()

        for filename in filenames:
            template_file = Path(dirpath).joinpath(filename)

            if not full and any(ignore_file in str(template_file) for ignore_file in IGNORED_FILES):
                continue

            with template_file.open(mode='r') as fr:
                with _dir.joinpath(Path(filename).name.replace(TEMPLATE_SUFFIX, '')).open(mode='w') as fw:
                    fw.write(fr.read())
