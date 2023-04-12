import logging
from datetime import datetime
from pathlib import Path

import httomo.globals


def setup_logger(out_dir: str):
    # Generate timestamped output directory
    httomo.globals.run_out_dir = _create_run_out_dir(out_dir)

    # Create empty `user.log` file
    user_log_path = httomo.globals.run_out_dir / "user.log"
    Path.touch(user_log_path)

    #: set up logging to a user.log file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        filename=f"{httomo.globals.run_out_dir}/user.log",
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    #: set up an easy format for console use
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    user_logger = logging.getLogger(__file__)
    user_logger.setLevel(logging.DEBUG)
    return user_logger


def _create_run_out_dir(dir: Path) -> Path:
    """Create the output directory in which the results of the httomo run are
    saved.
    """
    run_out_dir = dir.joinpath(f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output")
    Path.mkdir(run_out_dir)
    return run_out_dir
