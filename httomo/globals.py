from logging import Logger
import os
from pathlib import Path
from typing import Optional

run_out_dir: os.PathLike = Path('.')
logger: Optional[Logger] = None
gpu_id: int = -1
