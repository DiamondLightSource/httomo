from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from httomo.utils import log_exception


class Sweep(yaml.SafeLoader):
    """Class for representing a parameter sweep when explicitly given the values
    to sweep over. This will simply be a tuple of the given values.
    """

    def __init__(self, node):
        return tuple(self.construct_sequence(node))


class SweepRange(yaml.SafeLoader):
    """Class for representing a parameter sweep over a given range. This will be
    a tuple of values which are defined based on the start, stop, step values
    that are in the dict.
    """

    def __init__(self, node):
        # Form a dict from the YAML marked by `!SweepRange`
        range_dict = self.construct_mapping(node)
        # Check that only 3 keys are in the given dict/mapping: `start`, `stop`,
        # and `step`
        keys = range_dict.keys()
        is_valid_range = "start" in keys and "stop" in keys and "step" in keys
        if not is_valid_range:
            err_str = (
                "Please provide `start`, `stop`, `step` values when "
                "specifying a range to peform a parameter sweep over."
            )
            log_exception(err_str)
            raise ValueError(err_str)

        # Define the range based on the start, stop, step values
        param_vals = np.arange(
            range_dict["start"], range_dict["stop"], range_dict["step"]
        )
        param_vals = tuple(param_vals)
        return param_vals


def _get_loader():
    """Add Sweep and SweepRange constructors to PyYAML's SafeLoader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!Sweep", Sweep.__init__)
    loader.add_constructor("!SweepRange", SweepRange.__init__)
    return loader


def get_external_package_current_version(package: str) -> str:
    """
    Get current version of the external package
    from httomo/methods_database/packages/external/versions.yaml
    """
    versions_file = (
        Path(__file__).parent / "methods_database/packages/external/versions.yaml"
    )
    with open(versions_file, "r") as f:
        versions = yaml.safe_load(f)

    return str(versions[package]["current"][0])


def open_yaml_config(filepath: Path) -> List[Dict[str, Dict[str, Dict[str, Any]]]]:
    """Open and read a given YAML config file into a python data structure.

    Parameters
    ----------
    filepath : Path
        The file containing the YAML config.

    Returns
    -------
    List[Dict[str, Dict[str, Dict[str, Any]]]]
        A list of dicts, where each dict represents a task in the user config
        YAML file.
    """
    with open(filepath, "r") as f:
        conf = yaml.load(f, Loader=_get_loader())

    return conf
