from logging import Logger
import os
from pathlib import Path
from typing import Optional

run_out_dir: os.PathLike = Path('.')
logger: Optional[Logger] = None
gpu_id: int = -1
# maximum slices to use in CPU-only section
MAX_CPU_SLICES: int = 64
CHUNK_INTERMEDIATE: bool = False
COMPRESS_INTERMEDIATE: bool = False
