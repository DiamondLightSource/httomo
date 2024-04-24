import sys
from pathlib import Path

from loguru import logger


def setup_logger(out_path: Path):
    concise_logfile_path = out_path / "user.log"
    logger.remove(0)
    # Concise logs displayed in terminal
    logger.add(sink=sys.stdout, level="INFO", colorize=True, format="{message}")
    # Concise logs written to file
    logger.add(sink=concise_logfile_path, level="INFO", colorize=False, format="{message}")
