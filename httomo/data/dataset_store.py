from os import PathLike
from pathlib import Path
import time
import h5py
from typing import Literal, Optional, Tuple, Union
from httomo.data.hdf._utils.reslice import reslice
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import (
    DataSetSource,
    ReadableDataSetSink,
)
from mpi4py import MPI
import numpy as np
from numpy.typing import DTypeLike
import weakref

from httomo.utils import log_once, make_3d_shape_from_shape

"""
This is from the final handover call:

# Cases

- *We always process the data in blocks*

## 1 Process, all fits memory

- chunk_shape = global_shape   
- chunk_index = (0, 0, 0)
- the whole chunk is a numpy array, in a `DataSet` in-memory
- `write_block` -> writes a `DataSetBlock` into the chunk `DataSet`
- `read_block` -> get a block, which might be the full size of the `DataSet` for the chunk, 
  it's an in-memory slice of the chunk `DataSet`

### Reslice

- does nothing mostly
- updates `slicing_dim` somewhere, so that `read_block` knows how to slice the block

## 2 Processes, all fits in memory (or more)

(assume slicing dim is 0)

- chunk_shape < global_shape, e.g. `(5, 20, 30) < (10, 20, 30)`
- `chunk_index`: in rank 0: `(0, 0, 0)`, in rank 1: `(5, 0, 0)`
- the whole chunk is in numpy array, in a `DataSet`, in-memory
- BUT: each process as a part of the global data in-memory
- `write_block` -> writes a `DataSetBlock` into the chunk `DataSet`
- `read_block` -> get a block, which might be the full size of the `DataSet` for the chunk

### Reslice

- call the MPI memory-based reslice function we always had (using `MPI.allgather`)
- we have a new chunk in each process, wich chunk_shape=`(10, 10, 30)`
- we have `chunk_index`: in rank 0: `(0, 0, 0)`, in rank 1: `(0, 10, 0)`
- updates `slicing_dim` somewhere, so that `read_block` knows how to slice the block


## 2 Processes, doesn't memory (or more)

(assume slicing dim is 0)

- chunk_shape < global_shape, e.g. `(5, 20, 30) < (10, 20, 30)`
- `chunk_index`: in rank 0: `(0, 0, 0)`, in rank 1: `(5, 0, 0)`
- *the global data is fully is a single h5py file*
  - it is the whole global cube!
  - each process only needs to access a chunk, block-wise, out of that file
  - has the same interface as numpy array (`shape`, indexing, ...)
  - BUT: as soon as we index, we read the file into an in-memory numpy array
  - --> we cannot create a "view" of a subset of the file, referencing the disk
  - so, each process needs to keep track of the start of its chunk within the file, 
    and when read_block is called with start index 0, we add the chunk_index to that
- `write_block` -> writes a `DataSetBlock` into the file, with the correct offset (`chunk_index + block_index`)
- `read_block` -> get a block from the file, with offset `chunk_index + block_index`
- --> `FullFileDataSet` class takes care of that

### Reslice

- we have full globally shared file, all the data is there on disk already
- does nothing mostly
- updates `slicing_dim` somewhere, so that `read_block` knows how to slice the block
- update `chunk_index`: in rank 0: `(0, 0, 0)`, in rank 1: `(0, 10, 0)`


"""

# Notes:
# - refactoring the nested if into separate function is badly needed


class DataSetStoreWriter(ReadableDataSetSink):
    """A DataSetSink that can be used to store block-wise data in the current chunk (for the current process).

    It uses memory by default - but if there's a memory allocation error, a temporary h5 file is used
    to back the dataset's memory.

    The `make_reader` method can be used to create a DataSetStoreReader from this writer.
    It is intended to be used after the writer has finished, to read the data blockwise again.
    It will use the same underlying data store (h5 file or memory)"""

    def __init__(
        self,
        slicing_dim: Literal[0, 1, 2],
        comm: MPI.Comm,
        temppath: PathLike,
        memory_limit_bytes: int = 0,
    ):
        self._slicing_dim = slicing_dim
        self._comm = comm

        self._temppath = temppath
        self._readonly = False
        self._h5file: Optional[h5py.File] = None
        self._h5filename: Optional[Path] = None
        self._memory_limit_bytes: int = memory_limit_bytes

        self._data: Optional[Union[np.ndarray, h5py.Dataset]] = None

        self._global_shape: Optional[Tuple[int, int, int]] = None
        self._chunk_shape: Optional[Tuple[int, int, int]] = None
        self._global_index: Optional[Tuple[int, int, int]] = None

        # make sure finalize is called when this object is garbage-collected
        weakref.finalize(self, weakref.WeakMethod(self.finalize))

    @property
    def is_file_based(self) -> bool:
        return self._h5filename is not None

    @property
    def filename(self) -> Optional[Path]:
        return self._h5filename

    @property
    def comm(self) -> MPI.Comm:
        return self._comm

    # ??? do we need these properties?
    @property
    def global_shape(self) -> Tuple[int, int, int]:
        assert self._global_shape is not None
        return self._global_shape

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        assert self._chunk_shape is not None
        return self._chunk_shape

    @property
    def global_index(self) -> Tuple[int, int, int]:
        assert self._global_index is not None
        return self._global_index

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    @property
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data

    def write_block(self, block: DataSetBlock):
        if self._readonly:
            raise ValueError("Cannot write after creating a reader")
        block.to_cpu()
        start = max(block.chunk_index)
        if self._data is None:
            # if non-slice dims in block are different, update the shapes here
            self._global_shape = block.global_shape
            self._chunk_shape = block.chunk_shape
            self._global_index = (
                block.global_index[0] - block.chunk_index[0],
                block.global_index[1] - block.chunk_index[1],
                block.global_index[2] - block.chunk_index[2],
            )
            self._aux_data = block.aux_data
            self._create_new_data(block)
        else:
            assert self._global_shape is not None
            assert self._chunk_shape is not None
            assert self._global_index is not None
            if any(self._global_shape[i] != block.global_shape[i] for i in range(3)):
                raise ValueError(
                    "Attempt to write a block with inconsistent shape to existing data"
                )

            if any(self._chunk_shape[i] != block.chunk_shape[i] for i in range(3)):
                raise ValueError(
                    "Attempt to write a block with inconsistent shape to existing data"
                )

            if any(
                self._global_index[i] != block.global_index[i] - block.chunk_index[i]
                for i in range(3)
            ):
                raise ValueError(
                    "Attempt to write a block with inconsistent shape to existing data"
                )

        # insert the slice here
        assert self._data is not None  # after the above methods, this must be set
        start_idx = [0, 0, 0]
        start_idx[self._slicing_dim] = start
        if self.is_file_based:
            start_idx[self._slicing_dim] += self._global_index[self._slicing_dim]
        self._data[
            start_idx[0] : start_idx[0] + block.shape[0],
            start_idx[1] : start_idx[1] + block.shape[1],
            start_idx[2] : start_idx[2] + block.shape[2],
        ] = block.data

    def _get_global_h5_filename(self) -> PathLike:
        """Creates a temporary h5 file to back the storage (using nanoseconds timestamp
        for uniqueness).
        """
        filename = str(Path(self._temppath) / f"httom_tmpstore_{time.time_ns()}.hdf5")
        filename = self.comm.bcast(filename, root=0)

        self._h5filename = Path(filename)
        return self._h5filename

    def _create_new_data(self, block: DataSetBlock):
        # reduce memory errors across all processes - if any has a memory problem,
        # all should use a file
        sendBuffer = np.zeros(1, dtype=bool)
        recvBuffer = np.zeros(1, dtype=bool)
        try:
            self._data = self._create_numpy_data(self.chunk_shape, block.data.dtype)
        except MemoryError:
            sendBuffer[0] = True

        # do a logical or of all the memory errors across the processes
        self.comm.Allreduce([sendBuffer, MPI.BOOL], [recvBuffer, MPI.BOOL], MPI.LOR)

        if bool(recvBuffer[0]) is True:
            log_once(
                "Chunk does not fit in memory - using a file-based store",
                level=2,
            )
            # we create a full file dataset, i.e. file-based,
            # with the full global shape in it
            self._data = self._create_h5_data(
                self.global_shape,
                block.data.dtype,
                self._get_global_h5_filename(),
                self.comm,
            )

    def _create_numpy_data(
        self, chunk_shape: Tuple[int, int, int], dtype: DTypeLike
    ) -> np.ndarray:
        """Convenience method to enable mocking easily"""
        if (
            self._memory_limit_bytes > 0
            and np.prod(chunk_shape) * np.dtype(dtype).itemsize
            >= self._memory_limit_bytes
        ):
            raise MemoryError("Memory limit reached")

        return np.empty(chunk_shape, dtype)

    def _create_h5_data(
        self,
        global_shape: Tuple[int, int, int],
        dtype: DTypeLike,
        file: PathLike,
        comm: MPI.Comm,
    ) -> h5py.Dataset:
        """Creates a h5 data file based on the file-like object given.
        The returned data object behaves like a numpy array, so can be used freely within
        a DataSet."""

        self._h5file = h5py.File(file, "w", driver="mpio", comm=comm)
        # set how data should be chunked when saving
        # chunks = list(global_shape)
        # chunks[slicing_dim] = 1
        # TODO: best first measure performance impact
        # probably good chunk is (4, 4, <detector_x>)
        h5data = self._h5file.create_dataset("data", global_shape, dtype)

        return h5data

    def make_reader(
        self, new_slicing_dim: Optional[Literal[0, 1, 2]] = None
    ) -> DataSetSource:
        """Create a reader from this writer, reading from the same store"""
        if self._data is None:
            raise ValueError("Cannot make reader when no data has been written yet")
        self._readonly = True
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
        reader = DataSetStoreReader(self, new_slicing_dim)
        # make sure finalize is called when reader object is garbage-collected
        weakref.finalize(reader, weakref.WeakMethod(reader.finalize))
        return reader

    def finalize(self):
        self._data = None
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None


class DataSetStoreReader(DataSetSource):
    """Class to read from a store that has previously been written by DataSetStoreWriter,
    in a block-wise fashion.
    """

    def __init__(
        self, source: DataSetStoreWriter, slicing_dim: Optional[Literal[0, 1, 2]] = None
    ):
        self._comm = source.comm
        self._global_shape = source.global_shape
        self._global_index = source.global_index
        self._chunk_shape = source.chunk_shape
        self._aux_data = source.aux_data
        if source._data is None:
            raise ValueError(
                "Cannot create DataSetStoreReader when no data has been written"
            )

        self._h5file: Optional[h5py.File] = None
        self._h5filename: Optional[Path] = None
        source_data = source._data
        if source.is_file_based:
            self._h5filename = source.filename
            self._h5file = h5py.File(source.filename, "r")
            source_data = self._h5file["data"]

        if slicing_dim is None or slicing_dim == source.slicing_dim:
            self._slicing_dim = source.slicing_dim
            self._data = source_data
        else:
            self._data = self._reslice(source.slicing_dim, slicing_dim, source_data)
            self._slicing_dim = slicing_dim

        source.finalize()

    @property
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def is_file_based(self) -> bool:
        return self._h5filename is not None

    @property
    def filename(self) -> Optional[Path]:
        return self._h5filename

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._global_shape

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return self._chunk_shape

    @property
    def global_index(self) -> Tuple[int, int, int]:
        return self._global_index

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    def _reslice(
        self,
        old_slicing_dim: Literal[0, 1, 2],
        new_slicing_dim: Literal[0, 1, 2],
        data: Union[np.ndarray, h5py.Dataset],
    ) -> Union[np.ndarray, h5py.Dataset]:
        assert old_slicing_dim != new_slicing_dim
        if self._comm.size == 1:
            assert data.shape == self._global_shape
            assert self._comm.size == 1
            return data
        else:
            assert self._comm.size > 1
            if isinstance(data, np.ndarray):  # in-memory chunks
                # we only see a chunk, so we need to do MPI-based reslicing
                array, newdim, startidx = reslice(
                    data, old_slicing_dim + 1, new_slicing_dim + 1, self._comm
                )
                self._chunk_shape = array.shape  #  type: ignore
                assert newdim == new_slicing_dim + 1
                idx = [0, 0, 0]
                idx[new_slicing_dim] = startidx
                self._global_index = (idx[0], idx[1], idx[2])
                return array
            else:
                # we have a full, file-based dataset - all we have to do
                # is calculate the new chunk shape and start index
                rank = self._comm.rank
                nproc = self._comm.size
                length = self._global_shape[new_slicing_dim]
                startidx = round((length / nproc) * rank)
                stopidx = round((length / nproc) * (rank + 1))
                chunk_shape = list(self._global_shape)
                chunk_shape[new_slicing_dim] = stopidx - startidx
                self._chunk_shape = make_3d_shape_from_shape(chunk_shape)
                idx = [0, 0, 0]
                idx[new_slicing_dim] = startidx
                self._global_index = (idx[0], idx[1], idx[2])
                return data

    def read_block(self, start: int, length: int) -> DataSetBlock:
        start_idx = [0, 0, 0]
        start_idx[self._slicing_dim] = start
        if self.is_file_based:
            start_idx[self._slicing_dim] += self._global_index[self._slicing_dim]
        shape = list(self._global_shape)
        shape[self._slicing_dim] = length

        block_data = self._data[
            start_idx[0] : start_idx[0] + shape[0],
            start_idx[1] : start_idx[1] + shape[1],
            start_idx[2] : start_idx[2] + shape[2],
        ]

        block = DataSetBlock(
            data=block_data,
            aux_data=self._aux_data,
            slicing_dim=self._slicing_dim,
            block_start=start,
            chunk_start=self._global_index[self._slicing_dim],
            global_shape=self._global_shape,
            chunk_shape=self._chunk_shape,
        )

        return block

    def finalize(self):
        self._data = None
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
        # also delete the file
        if self._h5filename is not None and self._comm.rank == 0:
            self._h5filename.unlink()
            self._h5filename = None
