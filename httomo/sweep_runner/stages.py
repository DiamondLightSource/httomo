from typing import Any, List, NamedTuple, Tuple

from httomo.runner.method_wrapper import MethodWrapper


class NonSweepStage(NamedTuple):
    methods: List[MethodWrapper]


class SweepStage(NamedTuple):
    method: MethodWrapper
    param_name: str
    values: Tuple[Any, ...]


class Stages(NamedTuple):
    """
    Groups method wrappers into three stages:
    - wrappers representing methods that are before the sweep (excluding the loader)
    - wrapper representing the method in the sweep to execute multiple times
    - wrappers representing methods that are after the sweep
    """

    before_sweep: NonSweepStage
    sweep: SweepStage
    after_sweep: NonSweepStage
