from typing import List, Literal, Optional, Tuple

import numpy as np

from httomo.base_block import BaseBlock, generic_array
from httomo.block_interfaces import BlockIndexing
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.utils import make_3d_shape_from_array, make_3d_shape_from_shape


class DataSetBlock(BaseBlock, BlockIndexing):
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
        padding: Tuple[int, int] = (0, 0),
    ):
        """Constructs a data block for processing in the pipeline in high throughput runs.

        Parameters
        ----------

        data: ndarray
            A numpy or cupy array, 3D, holding the data represented by this block.
        aux_data: AuxiliaryData
            Object handling the flats, darks, and angles arrays
        slicing_dim: Literal[0, 1, 2]
            The slicing dimension in the global data that this block represents as slice of.
            This is to facilitate parallel processing - data is sliced in one of the 3 dimensions.
        block_start: int
            The index in slicing dimensions within the chunk that this block starts at. It is relative
            to the start of the chunk.
        chunk_start: int
            The index in slicing dimension within the global data that the underlying chunk starts.
            A chunk is a unit of the global data that is handled by a single MPI process, while a block
            might be a smaller part than the chunk.
        global_shape:  Optional[Tuple[int, int, int]]
            The shape of the global data across all processes. If not given, it assumes this block
            represents the full global data (no slicing done).
        chunk_shape: Optional[Tuple[int, int, int]]
            The shape of the chunk that this block belongs to. If not given, it assumes this block
            spans the full chunk.
        padding: Tuple[int, int]
            Padding information - holds the number of padded slices before and after the core area of the block,
            in slicing dimension. If not given, no padding is assumed.

            Note that the padding information should be added to the data's shape, i.e. block_start, chunk_start,
            chunk_shape, and the data's shape includes the padded slices. And therefore block_start or chunk_start
            may have negative values of up to -padding[0]. The global_shape is not adapted for padding.
        """
        super().__init__(data, aux_data)
        self._slicing_dim = slicing_dim
        self._block_start = block_start
        self._chunk_start = chunk_start
        self._padding = padding

        if global_shape is None:
            global_shape_t = list(data.shape)
            global_shape_t[slicing_dim] -= padding[0] + padding[1]
            self._global_shape = make_3d_shape_from_shape(global_shape_t)
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
        global_index[slicing_dim] += chunk_start + block_start + padding[0]
        self._global_index = make_3d_shape_from_shape(global_index)

        self._check_inconsistencies()

    def _check_inconsistencies(self):
        if self.padding[0] < 0 or self.padding[1] < 0:
            raise ValueError("padding values cannot be negative")
        if self.chunk_index[self.slicing_dim] + self._padding[0] < 0:
            raise ValueError("block start index must be >= 0")
        if (
            self.chunk_index[self.slicing_dim]
            + self.shape[self.slicing_dim]
            - self._padding[1]
            > self.chunk_shape[self.slicing_dim]
        ):
            raise ValueError("block spans beyond the chunk's boundaries")
        if self.global_index[self.slicing_dim] + self._padding[0] < 0:
            raise ValueError("chunk start index must be >= 0")
        if (
            self.global_index[self.slicing_dim]
            + self.shape[self.slicing_dim]
            - self._padding[1]
            > self.global_shape[self.slicing_dim]
        ):
            raise ValueError("chunk spans beyond the global data boundaries")
        if any(
            self.chunk_shape[i] > self.global_shape[i]
            for i in range(3)
            if i != self.slicing_dim
        ):
            raise ValueError(
                "chunk shape is larger than the global shape in non-slicing dimensions"
            )
        if (
            self.chunk_shape[self.slicing_dim] - self.padding[0] - self.padding[1]
            > self.global_shape[self.slicing_dim]
        ):
            raise ValueError(
                "chunk shape is larger than the global shape in slicing dimension"
            )
        if any(self.shape[i] > self.chunk_shape[i] for i in range(3)):
            raise ValueError("block shape is larger than the chunk shape")
        if any(
            self.shape[i] != self.global_shape[i]
            for i in range(3)
            if i != self.slicing_dim
        ):
            raise ValueError(
                "block shape inconsistent with non-slicing dims of global shape"
            )

        assert not any(
            self.chunk_shape[i] != self.global_shape[i]
            for i in range(3)
            if i != self.slicing_dim
        )

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
            == self.chunk_shape[self._slicing_dim] - self.padding[0]
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

    @data.deleter
    def data(self):
        del self._data
        del self._global_shape
        del self._chunk_shape

    @property
    def is_padded(self) -> bool:
        return self._padding != (0, 0)

    @property
    def padding(self) -> Tuple[int, int]:
        return self._padding

    @property
    def shape_unpadded(self) -> Tuple[int, int, int]:
        return self._correct_shape_for_padding(self.shape)

    @property
    def chunk_index_unpadded(self) -> Tuple[int, int, int]:
        return self._correct_index_for_padding(self.chunk_index)

    @property
    def chunk_shape_unpadded(self) -> Tuple[int, int, int]:
        return self._correct_shape_for_padding(self.chunk_shape)

    @property
    def global_index_unpadded(self) -> Tuple[int, int, int]:
        return self._correct_index_for_padding(self.global_index)

    def _correct_shape_for_padding(
        self, shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        if not self.padding:
            return shape
        shp = list(shape)
        shp[self.slicing_dim] -= self.padding[0] + self.padding[1]
        return make_3d_shape_from_shape(shp)

    def _correct_index_for_padding(
        self, index: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        if not self.padding:
            return index
        idx = list(index)
        idx[self.slicing_dim] += self.padding[0]
        return make_3d_shape_from_shape(idx)

    @property
    def data_unpadded(self) -> generic_array:
        if not self.padding:
            return self.data
        d = self.data
        slices = [slice(None), slice(None), slice(None)]
        slices[self.slicing_dim] = slice(
            self.padding[0], d.shape[self.slicing_dim] - self.padding[1]
        )
        return d[slices[0], slices[1], slices[2]]
