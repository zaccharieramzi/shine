import os
from pathlib import Path


LOGS_DIR = Path(os.environ.get('SHINE_LOGS', './logs'))
CHECKPOINTS_DIR = Path(os.environ.get('SHINE_CHECKPOINTS', '../new_models'))
DATA_DIR = Path(os.environ.get('SHINE_DATA', '../data'))
CONFIG_DIR = Path(os.environ.get('SHINE_CONFIG', '../experiments'))
IMAGENET_DIR = Path(os.environ.get('IMAGENET_DIR', '../data/imagenet'))
WORK_DIR = Path(os.environ.get('WORK_DIR', '~/workspace/shine'))
