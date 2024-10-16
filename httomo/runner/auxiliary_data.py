from typing import Optional, Tuple

import numpy as np

from httomo.types import generic_array
from httomo.utils import xp, gpu_enabled


class AuxiliaryData:
    """
    Keeps the darks and flats and angles auxiliary data together, and separate from the dataset.

    This allows them to be updated on their own without being affected by chunks and blocks, etc.,
    including the GPU/CPU transfers if needed.
    """

    def __init__(
        self,
        angles: np.ndarray,
        darks: Optional[np.ndarray] = None,
        flats: Optional[np.ndarray] = None,
    ):
        self._darks: Optional[generic_array] = darks
        self._flats: Optional[generic_array] = flats
        self._angles: np.ndarray = angles

    @property
    def darks_dtype(self) -> Optional[np.dtype]:
        return self._darks.dtype if self._darks is not None else None

    @property
    def darks_shape(self) -> Tuple[int, int, int]:
        if self._darks is None:
            return (0, 0, 0)
        assert len(self._darks.shape) == 3
        return (self._darks.shape[0], self._darks.shape[1], self._darks.shape[2])

    @property
    def flats_dtype(self) -> Optional[np.dtype]:
        return self._flats.dtype if self._flats is not None else None

    @property
    def flats_shape(self) -> Tuple[int, int, int]:
        if self._flats is None:
            return (0, 0, 0)
        assert len(self._flats.shape) == 3
        return (self._flats.shape[0], self._flats.shape[1], self._flats.shape[2])

    @property
    def angles_dtype(self) -> np.dtype:
        return self._angles.dtype

    @property
    def angles_length(self) -> int:
        return len(self._angles)

    def get_darks(self, gpu=False) -> Optional[generic_array]:
        return self._get_field("darks", gpu)

    def get_flats(self, gpu=False) -> Optional[generic_array]:
        return self._get_field("flats", gpu)

    def get_angles(self) -> np.ndarray:
        return self._angles

    def set_darks(self, darks: generic_array) -> None:
        self._darks = darks

    def set_flats(self, flats: generic_array) -> None:
        self._flats = flats

    def set_angles(self, angles: np.ndarray) -> None:
        assert getattr(angles, "device", None) is None, "Angles must be a CPU array"
        self._angles = angles

    def _get_field(self, field: str, gpu=False) -> generic_array:
        assert not gpu or gpu_enabled, "GPU can only be used if the GPU is enabled"

        array = getattr(self, f"_{field}")
        if array is None:
            return array

        # Note: if already on CPU/GPU, no copy is taken
        if gpu:
            array = xp.asarray(array)
        else:
            if xp.__name__ == "cupy":
                array = xp.asnumpy(array)

        setattr(self, f"_{field}", array)

        return array

    def drop_darks_flats(self):
        self._darks = None
        self._flats = None
