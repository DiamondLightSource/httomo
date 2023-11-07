from typing import Optional, TypeAlias, Union
from httomo.utils import gpu_enabled, xp
import numpy as np


class DataSet:
    """Holds the dataset the methods work on, handling both CPU or GPU.
    It handles the transfers if needed internally, given the currently-active
    GPU device in the CuPy context.

    Flats, darks, and angles are assumed to be read-only - separate copies on
    CPU and GPU are maintained as needed. Depending on where data itself is,
    the flats/darks/angles will be returned on the same device (and cached).
    
    Example::
       
       dataset = DataSet(data, angles, flats, darks)
       dataset.data        # access data
       dataset.to_gpu()    # transfer
       dataset.data        # now returns a GPU-based array
       assert dataset.is_gpu is True
    """

    generic_array: TypeAlias = Union[xp.ndarray, np.ndarray]

    # TODO: Think about whether detector_x, detector_y, angles_total should be here?

    def __init__(
        self,
        data: generic_array,
        angles: np.ndarray,
        flats: np.ndarray,
        darks: np.ndarray,
    ):
        self._angles = angles
        self._flats = flats
        self._darks = darks
        self._data = data
        self.lock()
        if gpu_enabled:
            import cupy as cp
            # cached GPU data - transferred lazily
            self._angles_gpu: Optional[cp.ndarray] = None
            self._flats_gpu: Optional[cp.ndarray] = None
            self._darks_gpu: Optional[cp.ndarray] = None
            # keep track if fields have been reset on GPU, to ensure
            # transferring back to CPU if needed
            self._angles_dirty: bool = False
            self._flats_dirty: bool = False
            self._darks_dirty: bool = False

    @property
    def angles(self) -> generic_array:
        return self._get_value("angles")

    @angles.setter
    def angles(self, new_data: generic_array):
        self._set_value("angles", new_data)

    @property
    def angles_radians(self) -> generic_array:
        """Alias for angles"""
        return self.angles

    @angles_radians.setter
    def angles_radians(self, new_data: generic_array):
        """Alias setter for angles"""
        self.angles = new_data

    @property
    def darks(self) -> generic_array:
        return self._get_value("darks")

    @darks.setter
    def darks(self, new_data: generic_array):
        self._set_value("darks", new_data)

    @property
    def dark(self) -> generic_array:
        """Alias for darks"""
        return self.darks

    @dark.setter
    def dark(self, new_data: generic_array):
        self.darks = new_data

    @property
    def flats(self) -> generic_array:
        return self._get_value("flats")

    @flats.setter
    def flats(self, new_data: generic_array):
        self._set_value("flats", new_data)

    @property
    def flat(self) -> generic_array:
        """Alias for flats"""
        return self.flats

    @flat.setter
    def flat(self, new_data: generic_array):
        self.flats = new_data

    @property
    def data(self) -> generic_array:
        return self._data

    @data.setter
    def data(self, new_data: generic_array):
        self._data = new_data

    @property
    def is_gpu(self) -> bool:
        """Check if arrays are currently residing on GPU"""
        if not gpu_enabled:
            return False
        return getattr(self._data, "device", None) is not None

    @property
    def is_cpu(self) -> bool:
        """Check if arrays are currently residing on CPU"""
        return not self.is_gpu

    def lock(self):
        """Makes angles, darks and flats read-only, to avoid coding errors.
        Note: this is the default in the constructor - only the the data array is writable
        Also note that this is only supported for the numpy arrays - gpu arrays can't be locked
        """
        self._angles.setflags(write=False)
        self._darks.setflags(write=False)
        self._flats.setflags(write=False)
        self._is_locked = True

    def unlock(self):
        """Unlock the read-only flag for the darks, angles, and flats members,
        allowing to overwrite them with the setter or modify their values"""
        self._angles.setflags(write=True)
        self._darks.setflags(write=True)
        self._flats.setflags(write=True)
        self._is_locked = False

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    def to_gpu(self):
        """Transfer dataset to GPU if not already."""
        if not gpu_enabled:
            raise ValueError("cannot transfer to GPU if not enabled")
        self._data = xp.asarray(self._data)

    def to_cpu(self):
        """Transfter dataset to CPU (if not already)"""
        if not self.is_gpu:
            return
        self._data = xp.asnumpy(self._data)

    def __dir__(self) -> list[str]:
        """Return only those properties that are relevant for the data"""
        return ["data", "angles", "angles_radians", "darks", "flats", "dark", "flat"]

    ###### internal helpers ######

    def _get_value(self, field: str) -> generic_array:
        """Helper function to get a field from this object.
        It uses getattr/setattr a lot, to allow for re-use from all the getters"""
        if self.is_gpu:
            setattr(
                self,
                f"_{field}_gpu",
                self._transfer_if_needed(
                    getattr(self, f"_{field}"), getattr(self, f"_{field}_gpu")
                ),
            )
            return getattr(self, f"_{field}_gpu")
        if gpu_enabled and getattr(self, f"_{field}_dirty"):
            setattr(self, f"_{field}", xp.asnumpy(getattr(self, f"_{field}_gpu")))
            setattr(self, f"_{field}_dirty", False)
        return getattr(self, f"_{field}")

    def _set_value(self, field: str, new_data: generic_array):
        """Sets a value of a field in this object, only if unlocked.
        It is a helper used in the setters for darks, flats, angles"""
        if self.is_locked:
            raise ValueError(f"attempt to reset {field} in a locked dataset")
        if not gpu_enabled:
            setattr(self, f"_{field}", new_data)
        if getattr(new_data, "device", None) is not None:
            # got GPU data - mark CPU dirty
            setattr(self, f"_{field}_gpu", new_data)
            setattr(self, f"_{field}_dirty", True)
        else:
            # got CPU data - make sure we remove cached GPU array
            setattr(self, f"_{field}_gpu", None)
            setattr(self, f"_{field}", new_data)

    def _transfer_if_needed(self, cpuarray: np.ndarray, gpuarray: Optional[xp.ndarray]):
        """Internal helper to transfer flats/darks/angles lazily"""
        if gpuarray is None:
            gpuarray = xp.asarray(cpuarray)
        assert (
            gpuarray.device.id == xp.cuda.Device().id
        ), f"GPU array is on a different GPU (expected: {xp.cuda.Device().id}, actual: {gpuarray.device.id})"
        return gpuarray
