import os
from pathlib import Path

run_out_dir: os.PathLike = Path(".")
gpu_id: int = -1
# maximum slices to use in CPU-only section
MAX_CPU_SLICES: int = (
    64  # A some random number which will be overwritten by --max-cpu_slices flag during runtime
)
FRAMES_PER_CHUNK: int = 1  # if given as 0, then write contiguous (no chunking)
INTERMEDIATE_FORMAT: str = "hdf5"
COMPRESS_INTERMEDIATE: bool = False
SYSLOG_SERVER = "localhost"
SYSLOG_PORT = 514
