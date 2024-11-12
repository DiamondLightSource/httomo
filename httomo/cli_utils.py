from pathlib import Path


MANUAL_SWEEP_TAG = "!Sweep"
RANGE_SWEEP_TAG = "!SweepRange"


def is_sweep_pipeline(file_path: Path) -> bool:
    """
    Determine if the given pipeline contains a parameter sweep
    """
    with open(file_path) as f:
        for line in f:
            if MANUAL_SWEEP_TAG in line or RANGE_SWEEP_TAG in line:
                return True
    return False
