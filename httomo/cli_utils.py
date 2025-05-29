import json
from pathlib import Path
from typing import Union, Dict, List, Any

MANUAL_SWEEP_TAG = "!Sweep"
RANGE_SWEEP_TAG = "!SweepRange"


def is_sweep_pipeline(pipeline: Union[Path, str]) -> bool:
    """
    Determine if the given pipeline contains a parameter sweep.

    Args:
        pipeline: Either a path to a YAML pipeline file, a JSON string,
                 or a Python object representing the pipeline configuration.

    Returns:
        bool: True if the pipeline contains a parameter sweep, False otherwise.
    """
    # Handle direct list/dict objects
    if isinstance(pipeline, (list, dict)):
        return _check_pipeline_object(pipeline)

    # If it's a string that looks like JSON (starts with '['), try parsing it as JSON first
    if isinstance(pipeline, str) and pipeline.strip().startswith("["):
        try:
            pipeline_data = json.loads(pipeline)
            return _check_pipeline_object(pipeline_data)
        except json.JSONDecodeError:
            # If JSON parsing fails, continue with other checks
            pass

    # Check if pipeline is a path to a file
    if isinstance(pipeline, (Path, str)) and (
        isinstance(pipeline, Path) or Path(pipeline).exists()
    ):
        with open(pipeline) as f:
            for line in f:
                if MANUAL_SWEEP_TAG in line or RANGE_SWEEP_TAG in line:
                    return True
        return False

    # For any other string, try parsing as JSON as a fallback
    if isinstance(pipeline, str):
        try:
            pipeline_data = json.loads(pipeline)
            return _check_pipeline_object(pipeline_data)
        except json.JSONDecodeError:
            # If it's not valid JSON, it's not a sweep pipeline
            return False

    # If we reach here, we don't know how to handle this input
    return False


def _check_pipeline_object(pipeline_data):
    """Check a pipeline object (list/dict) for sweep parameters"""
    # Check each step in the pipeline
    for step in pipeline_data:
        if isinstance(step, dict) and "parameters" in step:
            # Recursively check parameters for sweep patterns
            if _contains_sweep_parameter(step["parameters"]):
                return True
    return False


def _contains_sweep_parameter(params: Dict[str, Any]) -> bool:
    """
    Recursively check if parameters contain sweep patterns.

    Args:
        params: Dictionary of parameters to check

    Returns:
        bool: True if sweep pattern found, False otherwise
    """
    for key, value in params.items():
        # Check for SweepRange pattern (dict with start, stop, step)
        if isinstance(value, dict) and all(
            k in value for k in ["start", "stop", "step"]
        ):
            return True

        # Check for Sweep pattern (list of values)
        if isinstance(value, list) and not isinstance(value, str) and len(value) > 1:
            return True

    return False
