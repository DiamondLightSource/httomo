import graypy
import sys
from pathlib import Path

from loguru import logger

from httomo import globals


def setup_logger(out_path: Path):
    concise_logfile_path = out_path / "user.log"
    verbose_logfile_path = out_path / "debug.log"
    logger.remove(0)
    # Concise logs displayed in terminal
    logger.add(sink=sys.stdout, level="INFO", colorize=True, format="{message}")
    # Concise logs written to file
    logger.add(
        sink=concise_logfile_path, level="INFO", colorize=False, format="{message}"
    )
    # Verbose logs written to file
    logger.add(sink=verbose_logfile_path, level="DEBUG", colorize=False, enqueue=True)
    # Verbose logs sent to syslog server in GELF format
    syslog_handler = graypy.GELFTCPHandler(globals.SYSLOG_SERVER, globals.SYSLOG_PORT)
    logger.add(sink=syslog_handler, level="DEBUG", colorize=False)
