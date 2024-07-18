import importlib.util

from pathlib import Path
from typing import Any, Dict, List

from httomo.ui_layer import PipelineConfig


MANUAL_SWEEP_TAG = "!Sweep"
RANGE_SWEEP_TAG = "!SweepRange"
PYTHON_SCRIPT_MODULE_NAME = "python_script_module"


def is_sweep_pipeline(file_path: Path) -> bool:
    """
    Determine if the given pipeline contains a parameter sweep
    """
    extension = file_path.suffix.lower()
    if extension == ".yaml":
        return _does_yaml_pipeline_contain_sweep(file_path)
    elif extension == ".py":
        return _does_python_pipeline_contain_sweep(file_path)
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


def _does_python_pipeline_contain_sweep(file_path: Path) -> bool:
    """
    Check for tuple value in any method params
    """
    # NOTE: Used information in the following link:
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(PYTHON_SCRIPT_MODULE_NAME, file_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    pipeline_config: PipelineConfig = module.methods_to_list()

    all_methods_params: List[Dict[str, Any]] = [
        method["parameters"] for method in pipeline_config
    ]
    for params in all_methods_params:
        for value in params.values():
            if isinstance(value, tuple):
                return True
    return False
