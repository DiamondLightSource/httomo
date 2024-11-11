from pathlib import Path


MANUAL_SWEEP_TAG = "!Sweep"
RANGE_SWEEP_TAG = "!SweepRange"


def is_sweep_pipeline(file_path: Path) -> bool:
    """
    Determine if the given pipeline contains a parameter sweep
    """
    extension = file_path.suffix.lower()
    if extension == ".yaml":
        return _does_yaml_pipeline_contain_sweep(file_path)
    else:
        raise ValueError(f"Unrecognised pipeline file extension: {extension}")


def _does_yaml_pipeline_contain_sweep(file_path: Path) -> bool:
    """
    Check for `!Sweep` or `!SweepRange` tags
    """
    with open(file_path) as f:
        for line in f:
            if MANUAL_SWEEP_TAG in line or RANGE_SWEEP_TAG in line:
                return True
    return False
