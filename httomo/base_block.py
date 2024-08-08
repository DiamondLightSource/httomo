from typing import Tuple

import numpy as np

from httomo.block_interfaces import BlockData, BlockTransfer, generic_array
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.utils import gpu_enabled, make_3d_shape_from_array, xp


class BaseBlock(BlockData, BlockTransfer):
    """
    Base block class providing default implementations for the data
    transferring/getting/setting behaviour needed for a block type to be processed by
    implementors of `MethodWrapper`. Ie, this class provides default implementations for the
    `BlockTransfer` and `BlockData` protocols.

    Note that the data indexing behaviour described in `DataIndexing` is not implemented in
    this class. If the default implementations for data transferring/getting/setting in this
    class are acceptable: inherit from `BaseBlock`, override where necessary, and implement
    `DataIndexing` in order to implement the `Block` protocol.
    """

    def __init__(self, data: np.ndarray, aux_data: AuxiliaryData) -> None:
        self._data = data
        self._aux_data = aux_data

    def __dir__(self) -> list[str]:
        """Return only those properties that are relevant for the data"""
        return ["data", "angles", "angles_radians", "darks", "flats", "dark", "flat"]

    @property
    def data(self) -> generic_array:
        return self._data

    @data.setter
    def data(self, new_data: generic_array):
        self._data = new_data

    @property
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data

    def _empty_aux_array(self):
        empty_shape = list(self._data.shape)
        return np.empty_like(self._data, shape=empty_shape)

    @property
    def angles(self) -> np.ndarray:
        return self._aux_data.get_angles()

    @angles.setter
    def angles(self, new_angles: np.ndarray):
        self._aux_data.set_angles(new_angles)

    @property
    def angles_radians(self) -> np.ndarray:
        return self.angles

    @angles_radians.setter
    def angles_radians(self, new_angles: np.ndarray):
        self.angles = new_angles

    @property
    def darks(self) -> generic_array:
        darks = self._aux_data.get_darks(self.is_gpu)
        if darks is None:
            darks = self._empty_aux_array()
        return darks

    @darks.setter
    def darks(self, darks: generic_array):
        self._aux_data.set_darks(darks)

    # alias
    @property
    def dark(self) -> generic_array:
        return self.darks

    @dark.setter
    def dark(self, darks: generic_array):
        self.darks = darks

    @property
    def flats(self) -> generic_array:
        flats = self._aux_data.get_flats(self.is_gpu)
        if flats is None:
            flats = self._empty_aux_array()
        return flats

    @flats.setter
    def flats(self, flats: generic_array):
        self._aux_data.set_flats(flats)

    # alias
    @property
    def flat(self) -> generic_array:
        return self.flats

    @flat.setter
    def flat(self, flats: generic_array):
        self.flats = flats

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the data in this block"""
        return make_3d_shape_from_array(self._data)

    def to_gpu(self):
        if not gpu_enabled:
            raise ValueError("no GPU available")
        self._data = xp.asarray(self.data, order="C")

    def to_cpu(self):
        if not gpu_enabled:
            return
        self._data = xp.asnumpy(self.data, order="C")

    @property
    def is_gpu(self) -> bool:
        return not self.is_cpu

    @property
    def is_cpu(self) -> bool:
        return getattr(self._data, "device", None) is None
