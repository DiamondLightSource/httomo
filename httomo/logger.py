import logging
from pathlib import Path
import os
from typing import Optional

def setup_logger(out_dir: Optional[os.PathLike]):
    if out_dir is None:
        raise ValueError("out_dir has not been set")
    out_path = Path(out_dir)
    
    # Create timestamped output directory
    Path.mkdir(out_path)

    # Create empty `user.log` file
    user_log_path = out_path / "user.log"
    Path.touch(user_log_path)

    #: set up logging to a user.log file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        filename=f"{out_path}/user.log",
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
    
    logging.getLogger('httomo.runner').setLevel(logging.INFO)
    return user_logger
