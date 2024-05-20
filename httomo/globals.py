import os
from pathlib import Path

run_out_dir: os.PathLike = Path('.')
gpu_id: int = -1
# maximum slices to use in CPU-only section
MAX_CPU_SLICES: int = 64
SYSLOG_SERVER = "localhost"
SYSLOG_PORT = 514
