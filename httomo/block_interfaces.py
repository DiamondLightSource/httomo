from typing import Literal, Protocol, Tuple, TypeVar

import numpy as np

from httomo.types import generic_array
from httomo.runner.auxiliary_data import AuxiliaryData


class BlockTransfer(Protocol):
    """
    Data transferring behaviour required for a block type to be processed by implementors of
    `MethodWrapper`
    """

    def to_gpu(self): ...  # pragma: no cover

    def to_cpu(self): ...  # pragma: no cover

    @property
    def is_gpu(self) -> bool: ...  # pragma: no cover

    @property
    def is_cpu(self) -> bool: ...  # pragma: no cover


class BlockData(Protocol):
    """
    Data setting/getting behaviour required for a block type to be processed by implementors of
    `MethodWrapper`
    """

    def __dir__(self) -> list[str]:
        """
        Return at least the following properties: `data`, `angles`, `angles_radians`, `darks`,
        `flats`, `dark`, `flat`
        """
        ...  # pragma: no cover

    @property
    def data(self) -> generic_array: ...  # pragma: no cover

    @data.setter
    def data(self, new_data: generic_array): ...  # pragma: no cover

    @property
    def aux_data(self) -> AuxiliaryData: ...  # pragma: no cover

    @property
    def angles(self) -> np.ndarray: ...  # pragma: no cover

    @angles.setter
    def angles(self, new_angles: np.ndarray): ...  # pragma: no cover

    @property
    def angles_radians(self) -> np.ndarray: ...  # pragma: no cover

    @angles_radians.setter
    def angles_radians(self, new_angles: np.ndarray): ...  # pragma: no cover

    @property
    def darks(self) -> generic_array: ...  # pragma: no cover

    @darks.setter
    def darks(self, darks: generic_array): ...  # pragma: no cover

    # alias
    @property
    def dark(self) -> generic_array: ...  # pragma: no cover

    @dark.setter
    def dark(self, darks: generic_array): ...  # pragma: no cover

    @property
    def flats(self) -> generic_array: ...  # pragma: no cover

    @flats.setter
    def flats(self, flats: generic_array): ...  # pragma: no cover

    # alias
    @property
    def flat(self) -> generic_array: ...  # pragma: no cover

    @flat.setter
    def flat(self, flats: generic_array): ...  # pragma: no cover

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the data in this block"""
        ...  # pragma: no cover

    @property
    def shape_unpadded(self) -> Tuple[int, int, int]:
        """Shape of the core date in this block, with padding removed"""
        ...  # pragma: no cover


class BlockIndexing(Protocol):
    """
    Data indexing behaviour required for a block type to be processed by implementors of
    `MethodWrapper`
    """

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        """The index of this block within the chunk handled by the current process"""
        ...  # pragma: no cover

    @property
    def chunk_index_unpadded(self) -> Tuple[int, int, int]:
        """The index of the core area of this block within the chunk (padding removed)"""
        ...  # pragma: no cover

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        """Shape of the full chunk handled by the current process"""
        ...  # pragma: no cover

    @property
    def chunk_shape_unpadded(self) -> Tuple[int, int, int]:
        """Shape of the full chunk core area handled by the current process (with padding removed)"""
        ...  # pragma: no cover

    @property
    def global_index(self) -> Tuple[int, int, int]:
        """The index of this block within the global data across all processes"""
        ...  # pragma: no cover

    @property
    def global_index_unpadded(self) -> Tuple[int, int, int]:
        """The index of the core area of this block within the global data across all processes (with padding removed)"""
        ...  # pragma: no cover

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        """Shape of the global data across all processes"""
        ...  # pragma: no cover

    @property
    def is_last_in_chunk(self) -> bool:
        """
        Check if the current dataset is the final one for the chunk handled by the current
        process
        """
        ...  # pragma: no cover

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        """Return the slicing dimenions of the block"""
        ...  # pragma: no cover

    @property
    def is_padded(self) -> bool:
        """Determine if this is a padded block"""
        ...  # pragma: no cover

    @property
    def padding(self) -> Tuple[int, int]:
        """Get the 'before' and 'after' padding values for this block.
        This is to be understood as the the number of padding slices in the 'slicing_dim' direction
        that come before and after the core area of the block.
        If no padding is used, it returns (0, 0).
        """
        ...  # pragma: no cover

    @property
    def data_unpadded(self) -> generic_array:
        """Return the data, but with the padding slices removed"""
        ...  # pragma: no cover


class Block(BlockData, BlockIndexing, BlockTransfer, Protocol):
    """
    All behaviour required for a block type to be processed by implementors of
    `MethodWrapper`
    """

    ...  # pragma: no cover


# The type parameter `T` is defined for use in any method signature that takes in an
# implementor of `Block` as a parameter and returns the *same* implementor of `Block` (such as
# the `execute()` method signature in `MethodWrapper`)
T = TypeVar("T", bound=Block)
