from typing import Optional, Tuple, Union
from typing_extensions import TypeAlias
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.utils import gpu_enabled, xp
import numpy as np

from httomo.utils import make_3d_shape_from_shape
from httomo.utils import make_3d_shape_from_array


class DataSetBlock:
    """Represents a slice/block of a dataset, as returned returned by `make_block`
    in a DataSet object. It is a DataSet (inherits from it) and users can mostly
    ignore the fact that it's just a view.

    It stores the base object internally and routes all calls for the auxilliary
    arrays to the base object (darks/flats/angles). It does not store these directly.
    """
    
    generic_array: TypeAlias = Union[np.ndarray, xp.ndarray]

    def __init__(
        self,
        data: np.ndarray,
        aux_data: AuxiliaryData,
        slicing_dim: int = 0,
        block_start: int = 0,
        chunk_start: int = 0,
        global_shape: Optional[Tuple[int, int, int]] = None,
        chunk_shape: Optional[Tuple[int, int, int]] = None,
    ):
        self._data = data
        self._aux_data = aux_data
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
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the data in this block"""
        return make_3d_shape_from_array(self._data)

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
    def is_cpu(self) -> bool:
        return getattr(self._data, "device", None) is None
    
    @property
    def is_gpu(self) -> bool:
        return not self.is_cpu
    
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
    def is_last_in_chunk(self) -> bool:
        """Check if the current dataset is the final one for the chunk handled by the current process"""
        return (
            self.chunk_index[self._slicing_dim] + self.shape[self._slicing_dim]
            == self.chunk_shape[self._slicing_dim]
        )

    @property
    def slicing_dim(self) -> int:
        return self._slicing_dim
    
    def _empty_aux_array(self):
        empty_shape = list(self._data.shape)
        empty_shape[self.slicing_dim] = 0
        return np.empty_like(self._data, shape=empty_shape)

    @property
    def data(self) -> generic_array:
        return self._data

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

    def to_gpu(self):
        if not gpu_enabled:
            raise ValueError("no GPU available")
        # from doc: if already on GPU, no copy is taken
        self._data = xp.asarray(self.data, order="C")        

    def to_cpu(self):
        if not gpu_enabled:
            return
        self._data = xp.asnumpy(self.data, order="C")
    
    def __dir__(self) -> list[str]:
        """Return only those properties that are relevant for the data"""
        return ["data", "angles", "angles_radians", "darks", "flats", "dark", "flat"]
<<<<<<< HEAD

    ###### internal helpers ######

    def get_value(
        self, field: str, data_is_gpu: Optional[bool] = None
    ) -> generic_array:
        """Helper function to get a field from this object.
        It uses getattr/setattr a lot, to allow for re-use from all the getters.

        `data_is_gpu` can be used to tell this method to assume the data array
        is on GPU or not - it will be used instead of self.is_gpu if given"""

        # angles always stay on CPU
        if field == "angles":
            return getattr(self, f"_{field}")

        is_gpu = data_is_gpu if data_is_gpu is not None else self.is_gpu
        if is_gpu:
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

    def set_value(self, field: str, new_data: generic_array):
        """Sets a value of a field in this object, only if unlocked.
        It is a helper used in the setters for darks, flats, angles"""
        if self.is_locked:
            raise ValueError(f"attempt to reset {field} in a locked dataset")
        if not gpu_enabled or field == "angles":
            assert (
                getattr(new_data, "device", None) is None
            ), f"GPU array for CPU-only field {field}"
            setattr(self, f"_{field}", new_data)
            return
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
            gpuarray = xp.asarray(cpuarray, order='C')
        assert (
            gpuarray.device.id == xp.cuda.Device().id
        ), f"GPU array is on a different GPU (expected: {xp.cuda.Device().id}, actual: {gpuarray.device.id})"
        return gpuarray


class DataSetBlock(DataSet):
    """Represents a slice/block of a dataset, as returned returned by `make_block`
    in a DataSet object. It is a DataSet (inherits from it) and users can mostly
    ignore the fact that it's just a view.

    It stores the base object internally and routes all calls for the auxilliary
    arrays to the base object (darks/flats/angles). It does not store these directly.
    """

    def __init__(self, base: DataSet, dim: int, start: int, length: int):
        idx_expr = [(0, base.shape[0]), (0, base.shape[1]), (0, base.shape[2])]
        idx_expr[dim] = (start, start + length)
        global_index = list(base.global_index)
        global_index[dim] += start
        # we pass an empty size-0 array to base class, as we're not going to use these
        # fields anyway here (we access the originals via self._base)
        super().__init__(
            data=base.get_data_block(
                idx_expr[0][0],
                idx_expr[0][1],
                idx_expr[1][0],
                idx_expr[1][1],
                idx_expr[2][0],
                idx_expr[2][1],
            ),
            flats=np.empty((0,)),
            darks=np.empty((0,)),
            angles=np.empty((0,)),
            global_shape=base.global_shape,
            global_index=(global_index[0], global_index[1], global_index[2]),
        )
        self._base = base
        idx = [0, 0, 0]
        idx[dim] = start
        self._chunk_shape = base.chunk_shape
        self._chunk_index = (idx[0], idx[1], idx[2])
        self._dim = dim

    @property
    def is_block(self) -> bool:
        return True

    @property
    def is_full(self) -> bool:
        """Check if the dataset is the full global data"""
        return False

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        """The index of this dataset within the chunk handled by the current process"""
        return self._chunk_index

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        """Shape of the full chunk handled by the current process"""
        return self._chunk_shape

    @property
    def data(self) -> DataSet.generic_array:
        return super().data

    @data.setter
    def data(self, new_data: DataSet.generic_array):
        super()._set_data(new_data)
        chunk_shape = np.array(new_data.shape)
        chunk_shape[self._dim] = self._base.chunk_shape[self._dim]
        self._chunk_shape = (chunk_shape[0], chunk_shape[1], chunk_shape[2])

    @property
    def is_last_in_chunk(self) -> bool:
        """Check if the current dataset is the final one for the chunk handled by the current process"""
        return (
            self.chunk_index[self._dim] + self.shape[self._dim]
            == self.chunk_shape[self._dim]
        )

    @property
    def base(self) -> DataSet:
        """Get the original (unblocked) dataset"""
        return self._base

    def set_value(self, field: str, new_data: DataSet.generic_array):
        raise ValueError(f"Cannot update field {field} in a block/slice dataset")

    def get_value(
        self, field: str, is_gpu: Optional[bool] = None
    ) -> DataSet.generic_array:
        return self._base.get_value(field, self.is_gpu if is_gpu is None else is_gpu)
    
    @property
    def has_gpu_darks(self) -> bool:
        return self._base.has_gpu_darks
    
    @property
    def has_gpu_flats(self) -> bool:
        return self._base.has_gpu_flats

    def make_block(self, dim: int, start: int = 0, length: Optional[int] = None):
        raise ValueError("Cannot slice a dataset that is already a slice")


class FullFileDataSet(DataSet):
    generic_array = DataSet.generic_array

    def __init__(
        self,
        data: np.ndarray,
        angles: np.ndarray,
        flats: np.ndarray,
        darks: np.ndarray,
        global_index: Tuple[int, int, int],
        chunk_shape: Tuple[int, int, int],
        shape: Tuple[int, int, int],
        data_offset: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__(
            data,
            angles,
            flats,
            darks,
            (data.shape[0], data.shape[1], data.shape[2]),
            global_index,
        )
        self._chunk_shape = chunk_shape
        self._shape = shape
        self._data_offset = data_offset

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return self._chunk_shape

    @property
    def is_full(self) -> bool:
        return True

    @property
    def data(self) -> DataSet.generic_array:
        # Note: this view doesn't allow write-back, it's a copy of the data in the file
        return self._data[
            self._global_index[0] : self._global_index[0] + self._chunk_shape[0],
            self._global_index[1] : self._global_index[1] + self._chunk_shape[1],
            self._global_index[2] : self._global_index[2] + self._chunk_shape[2],
        ]

    @data.setter
    def data(self, new_data: DataSet.generic_array):
        self.set_data_block((0, 0, 0), new_data)

    def set_data_block(
        self, start_idx: Tuple[int, int, int], new_data: DataSet.generic_array
    ):
        if any(
            start + length > self._chunk_shape[i]
            for i, (start, length) in enumerate(zip(start_idx, new_data.shape))
        ):
            raise ValueError(
                "in a FullFileDataSet, changing shape of a chunk is not allowed"
            )
        if self._data.dtype != new_data.dtype:
            raise ValueError(
                "in a FullFileDataSet, changing the datatype is not allowed"
            )
        if getattr(new_data, "device", None) is not None:
            new_data = xp.asnumpy(new_data)

        self._data[
            self._global_index[0]
            + start_idx[0] : self._global_index[0]
            + start_idx[0]
            + new_data.shape[0],
            self._global_index[1]
            + start_idx[1] : self._global_index[1]
            + start_idx[1]
            + new_data.shape[1],
            self._global_index[2]
            + start_idx[2] : self._global_index[2]
            + start_idx[2]
            + new_data.shape[2],
        ] = new_data

    def get_data_block(
        self, start0: int, stop0: int, start1: int, stop1: int, start2: int, stop2: int
    ) -> DataSet.generic_array:
        # `self._data_offset` and `self._global_index` are used for block reading offsets
        # required by `StandardTomoLoader`. For all other objects using `FullFileDataSet`,
        # `self.global_index` exclusively is used for offsetting block reads.
        start0 += self._global_index[0] + self._data_offset[0]
        stop0 += self._global_index[0] + self._data_offset[0]
        start1 += self._global_index[1] + self._data_offset[1]
        stop1 += self._global_index[1] + self._data_offset[1]
        start2 += self._global_index[2] + self._data_offset[2]
        stop2 += self._global_index[2] + self._data_offset[2]
        return self._data[start0:stop0, start1:stop1, start2:stop2]
=======
>>>>>>> feature/transparent-file-store
