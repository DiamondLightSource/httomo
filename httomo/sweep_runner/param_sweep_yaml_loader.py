from typing import Tuple, Union

import yaml
import numpy as np

from httomo.utils import log_exception


class ParamSweepYamlLoader(yaml.SafeLoader):
    """
    Load an HTTomo pipeline YAML file containing a parameter sweep, via the `!Sweep` or
    `!SweepRange` tag.
    """

    def sweep_manual(self, node) -> Tuple[Union[int, float], ...]:
        """
        A parameter sweep defined by explicitly given values to sweep over.
        This will simply be a tuple of the given values.
        """
        return tuple(self.construct_sequence(node))

    def sweep_range(self, node) -> Tuple[Union[int, float], ...]:
        """
        A parameter sweep defined by a given range. This will be a tuple of
        values which are defined based on the start, stop, step values that are
        in the dict.
        """
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
        return tuple(param_vals)


def get_param_sweep_yaml_loader() -> type[ParamSweepYamlLoader]:
    loader = ParamSweepYamlLoader
    loader.add_constructor("!Sweep", ParamSweepYamlLoader.sweep_manual)
    loader.add_constructor("!SweepRange", ParamSweepYamlLoader.sweep_range)
    return loader
