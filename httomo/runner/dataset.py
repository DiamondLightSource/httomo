from typing import Literal, Optional, Tuple

import numpy as np

from httomo.base_block import BaseBlock, generic_array
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.utils import make_3d_shape_from_array, make_3d_shape_from_shape


class DataSetBlock(BaseBlock):
    """
    Data storage type for block processing in high throughput runs
    """

    def __init__(
        self,
        data: np.ndarray,
        aux_data: AuxiliaryData,
        slicing_dim: Literal[0, 1, 2] = 0,
        block_start: int = 0,
        chunk_start: int = 0,
        global_shape: Optional[Tuple[int, int, int]] = None,
        chunk_shape: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__(data, aux_data)
        self._slicing_dim = slicing_dim
        self._block_start = block_start
        self._chunk_start = chunk_start

        if global_shape is None:
            self._global_shape = make_3d_shape_from_array(data)
        else:
            self._global_shape = global_shape
            
        if chunk_shape is None:
            self._chunk_shape = make_3d_shape_from_array(data)
        else:
            self._chunk_shape = chunk_shape

        chunk_index = [0, 0, 0]
        chunk_index[slicing_dim] += block_start
        self._chunk_index = make_3d_shape_from_shape(chunk_index)
        global_index = [0, 0, 0]
        global_index[slicing_dim] += chunk_start + block_start
        self._global_index = make_3d_shape_from_shape(global_index)

        self._check_inconsistencies()
        
    def _check_inconsistencies(self):
        if self.chunk_index[self.slicing_dim] < 0:
            raise ValueError("block start index must be >= 0")
        if self.chunk_index[self.slicing_dim] + self.shape[self.slicing_dim] > self.chunk_shape[self.slicing_dim]:
            raise ValueError("block spans beyond the chunk's boundaries")
        if self.global_index[self.slicing_dim] < 0:
            raise ValueError("chunk start index must be >= 0")
        if self.global_index[self.slicing_dim] + self.shape[self.slicing_dim] > self.global_shape[self.slicing_dim]:
            raise ValueError("chunk spans beyond the global data boundaries")
        if any(self.chunk_shape[i] > self.global_shape[i] for i in range(3)):    
            raise ValueError("chunk shape is larger than the global shape")
        if any(self.shape[i] > self.chunk_shape[i] for i in range(3)):
            raise ValueError("block shape is larger than the chunk shape")
        if any(self.shape[i] != self.global_shape[i] for i in range(3) if i != self.slicing_dim):
            raise ValueError("block shape inconsistent with non-slicing dims of global shape")
        
        assert not any(self.chunk_shape[i] != self.global_shape[i] for i in range(3) if i != self.slicing_dim)

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        """The index of this block within the chunk handled by the current process"""
        return self._chunk_index

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        """Shape of the full chunk handled by the current process"""
        return self._chunk_shape
    
    @property
    def global_index(self) -> Tuple[int, int, int]:
        """The index of this block within the global data across all processes"""
        return self._global_index

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        """Shape of the global data across all processes"""
        return self._global_shape
    
    @property
    def is_last_in_chunk(self) -> bool:
        """Check if the current dataset is the final one for the chunk handled by the current process"""
        return (
            self.chunk_index[self._slicing_dim] + self.shape[self._slicing_dim]
            == self.chunk_shape[self._slicing_dim]
        )

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim
    
    def _empty_aux_array(self):
        empty_shape = list(self._data.shape)
        empty_shape[self.slicing_dim] = 0
        return np.empty_like(self._data, shape=empty_shape)

    @property
    def data(self) -> generic_array:
        return super().data

    @data.setter
    def data(self, new_data: generic_array):
        global_shape = list(self._global_shape)
        chunk_shape = list(self._chunk_shape)
        for i in range(3):
            if i != self.slicing_dim:
                global_shape[i] = new_data.shape[i]
                chunk_shape[i] = new_data.shape[i]
            elif self._data.shape[i] != new_data.shape[i]:
                raise ValueError("shape mismatch in slicing dimension")
                
        self._data = new_data
        self._global_shape = make_3d_shape_from_shape(global_shape)
        self._chunk_shape = make_3d_shape_from_shape(chunk_shape)
