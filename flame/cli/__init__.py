import os
import sys

from .cli import cli
from .init import init
from .run import run

if sys.path[0] != os.getcwd():
    sys.path.insert(0, os.getcwd())
