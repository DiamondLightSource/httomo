from typing import List, NamedTuple

from httomo.runner.method_wrapper import MethodWrapper


class Stages(NamedTuple):
    """
    Groups method wrappers into three stages:
    - wrappers representing methods that are before the sweep (excluding the loader)
    - wrapper representing the method in the sweep to execute multiple times
    - wrappers representing methods that are after the sweep
    """

    before_sweep: List[MethodWrapper]
    sweep: MethodWrapper
    after_sweep: List[MethodWrapper]
