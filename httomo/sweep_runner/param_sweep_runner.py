from typing import Optional

from httomo.runner.dataset import DataSetBlock


class ParamSweepRunner:
    def __init__(self) -> None:
        self._block: Optional[DataSetBlock] = None

    @property
    def block(self) -> DataSetBlock:
        if self._block is None:
            raise ValueError("Block from input data has not yet been loaded")
        return self._block
