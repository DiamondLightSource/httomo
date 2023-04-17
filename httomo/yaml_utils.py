import yaml
import numpy as np
from typing import List, Dict
from pathlib import Path

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


def open_yaml_config(filepath: Path) -> List[Dict]:
    """Open and read a given YAML config file into a python data structure.

    Parameters
    ----------
    filepath : Path
        The file containing the YAML config.

    Returns
    -------
    List[Dict]
        A list of dicts, where each dict represents a task in the user config
        YAML file.
    """
    with open(filepath, "r") as f:
        conf = yaml.load(f, Loader=_get_loader())

    # TODO: validate the YAML to ensure there are no missing fields that are
    # required

    # TODO: perform parameter type checks

    return conf


def validate_yaml_config() -> bool:
    """Check that the pipeline YAML config isn't missing any required fields.

    Parameters
    ----------

    Returns
    -------
    bool
    """
    pass


def check_param_types() -> bool:
    """Perform type-checking of the values provided in the pipeline YAML config.

    Parameters
    ----------

    Returns
    -------
    """
    pass


def generate_conf_from_prerun() -> None:
    """Create a YAML config file based on a pre-run of given data

    Parameters
    ----------
    """
    pass
