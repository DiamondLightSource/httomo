from typing import List, NamedTuple

from httomo.runner.method_wrapper import MethodWrapper


class Stages(NamedTuple):
    """
    Groups method wrappers into three stages:
    - wrappers representing methods that are before the sweep (excluding the loader)
    - wrappers representing the sweep
    - wrappers representing methods that are after the sweep
    """

    before_sweep: List[MethodWrapper]
    sweep: List[MethodWrapper]
    after_sweep: List[MethodWrapper]
