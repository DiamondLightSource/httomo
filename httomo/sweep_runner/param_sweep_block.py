from typing import Tuple

import numpy as np

from httomo.base_block import BaseBlock
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.utils import make_3d_shape_from_array


class ParamSweepBlock(BaseBlock):
    """
    Data storage type for block processing in parameter sweep runs
    """

    def __init__(
        self,
        data: np.ndarray,
        aux_data: AuxiliaryData,
    ) -> None:
        super().__init__(data, aux_data)

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        return (0, 0, 0)

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return make_3d_shape_from_array(self._data)

    @property
    def global_index(self) -> Tuple[int, int, int]:
        return (0, 0, 0)

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self.chunk_shape

    @property
    def is_last_in_chunk(self) -> bool:
        return True
