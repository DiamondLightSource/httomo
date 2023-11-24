from io import BufferedRandom
from os import PathLike
import h5py
from tempfile import TemporaryFile, mkstemp
from typing import Literal, Optional, Tuple
from httomo.runner.dataset import DataSet, DataSetBlock
from httomo.runner.dataset_store_interfaces import DataSetSink, DataSetSource
from mpi4py import MPI
import numpy as np
from numpy.typing import DTypeLike


class DataSetStoreWriter(DataSetSink):
    """A DataSetSink that can be used to store block-wise data in the current chunk (for the current process).
    
    It uses memory by default - but if there's a memory allocation error, a temporary h5 file is used
    to back the dataset's memory. 
    
    The `make_reader` method can be used to create a DataSetStoreReader from this writer. 
    It is intended to be used after the writer has finished, to read the data blockwise again.
    It will use the same underlying data store (h5 file or memory)"""
    
    def __init__(
        self,
        full_size: int,
        slicing_dim: Literal[0, 1, 2],
        other_dims: Tuple[int, int],
        chunk_size: int,
        chunk_start: int,
        comm: MPI.Comm,
        temppath: PathLike,
    ):
        self._slicing_dim = slicing_dim
        self._comm = comm

        full = list(other_dims)
        full.insert(self._slicing_dim, full_size)
        chunk = list(other_dims)
        chunk.insert(self._slicing_dim, chunk_size)
        idx = [0, 0]
        idx.insert(self._slicing_dim, chunk_start)
        self._full_shape: Tuple[int, int, int] = tuple(full)  # type: ignore
        self._chunk_shape: Tuple[int, int, int] = tuple(chunk)  # type: ignore
        self._chunk_idx: Tuple[int, int, int] = tuple(idx)  # type: ignore
        self._temppath = temppath

        self._data: Optional[DataSet] = None

    @property
    def comm(self) -> MPI.Comm:
        return self._comm

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._full_shape

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return self._chunk_shape

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        return self._chunk_idx

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    def write_block(self, dataset: DataSetBlock):
        dataset.to_cpu()
        start = max(dataset.chunk_index)
        if self._data is None:
            block_shape = list(dataset.data.shape)
            block_shape.pop(self._slicing_dim)
            fullshape = block_shape.copy()
            fullshape.insert(self._slicing_dim, self._full_shape[self.slicing_dim])
            self._full_shape = tuple(fullshape)  # type: ignore
            chunkshape = block_shape.copy()
            chunkshape.insert(self._slicing_dim, self._chunk_shape[self._slicing_dim])
            self._chunk_shape = tuple(chunkshape)  # type: ignore
            self._create_new_data(dataset)
        else:
            if any(
                a != b
                for i, (a, b) in enumerate(zip(self._data.shape, dataset.shape))
                if i != self._slicing_dim
            ):
                raise ValueError(
                    "Attempt to write a block with inconsistent shape to existing data"
                )

        if (
            start + dataset.shape[self._slicing_dim]
            > self.chunk_shape[self._slicing_dim]
        ):
            raise ValueError("writing a block that is outside the chunk dimension")

        # insert the slice here
        assert self._data is not None  # after the above methods, this must be set
        to_idx = [slice(None), slice(None), slice(None)]
        to_idx[self._slicing_dim] = slice(
            start, start + dataset.shape[self._slicing_dim]
        )
        self._data.data[tuple(to_idx)] = dataset.data

    def _make_h5_file(self) -> BufferedRandom:
        """Creates a temporary h5 file to back the storage. On Linux, note that the 
        file is not visible in the filesystem - TemporaryFile opens it and immediate
        unlinks it."""
        tmp =  TemporaryFile(dir=str(self._temppath), suffix=".hdf5")
        return tmp

    def _create_new_data(self, dataset):
        try:
            data = self._create_numpy_data(tuple(self._chunk_shape), dataset.data.dtype)
        except MemoryError:
            data = self._create_h5_data(tuple(self._chunk_shape), dataset.data.dtype, 
                                        self._make_h5_file())
        self._data = DataSet(
            data=data,
            angles=dataset.angles,
            flats=dataset.flats,
            darks=dataset.darks,
            global_shape=self._full_shape,
            global_index=self.chunk_index,
        )

    @classmethod
    def _create_numpy_data(cls, chunk_shape: Tuple[int, int, int], dtype: DTypeLike) -> np.ndarray:
        """Convenience method to enable mocking easily"""
        return np.empty(chunk_shape, dtype)

    @classmethod
    def _create_h5_data(
        cls, chunk_shape: Tuple[int, int, int], dtype: DTypeLike, file: BufferedRandom
    ) -> np.ndarray:
        """Creates a h5 data file based on the file-like object given. 
        The returned data object behaves like a numpy array, so can be used freely within
        a DataSet."""
        
        f = h5py.File(file, 'w')
        data = f.create_dataset("data", chunk_shape, dtype)
        return data

    def make_reader(
        self, new_slicing_dim: Optional[int] = None
    ) -> "DataSetStoreReader":
        """Create a reader from this writer, reading from the same store"""
        return DataSetStoreReader(self, new_slicing_dim)

    def finalise(self):
        pass


class DataSetStoreReader(DataSetSource):
    """Class to read from a store that has previously been written by DataSetStoreWriter, 
    in a block-wise fashion.
    """
    
    def __init__(self, source: DataSetStoreWriter, slicing_dim: Optional[int] = None):
        self._comm = source.comm
        self._full_shape = source.global_shape
        self._chunk_idx = source.chunk_index
        self._chunk_shape = source.chunk_shape
        if source._data is None:
            raise ValueError(
                "Cannot create DataSetStoreReader when no data has been written"
            )
        self._data = source._data

        if slicing_dim is None or slicing_dim == source.slicing_dim:
            self._slicing_dim = source.slicing_dim
        else:
            self._reslice(slicing_dim)
        
        source.finalise()

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._full_shape

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return self._chunk_shape

    @property
    def chunk_index(self) -> Tuple[int, int, int]:
        return self._chunk_idx

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    def _reslice(self, slicing_dim: int):
        raise NotImplementedError("reslicing is not yet implemented")

    def read_block(self, start: int, length: int) -> DataSetBlock:
        block = self._data.make_block(self._slicing_dim, start, length)
        
        return block
