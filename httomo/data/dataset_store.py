from os import PathLike
from pathlib import Path
import time
import h5py
from tempfile import NamedTemporaryFile, TemporaryFile, mkstemp
from typing import IO, BinaryIO, Literal, Optional, Tuple
from httomo.data.hdf._utils.reslice import reslice
from httomo.runner.dataset import DataSet, DataSetBlock, FullFileDataSet
from httomo.runner.dataset_store_interfaces import DataSetSink, DataSetSource
from mpi4py import MPI
import numpy as np
from numpy.typing import DTypeLike
import weakref

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
# - if MemoryError is thrown in one process, all others should also use the file
# - refactoring the nested if into separate function is badly needed


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
        self._readonly = False
        self._h5file: Optional[h5py.File] = None
        self._h5filename: Optional[Path] = None

        self._data: Optional[DataSet] = None
        
        # make sure finalize is called when this object is garbage-collected
        weakref.finalize(self, self.finalize)

    @property
    def is_file_based(self) -> bool:
        return self._h5filename is not None
    
    @property
    def filename(self) -> Optional[Path]:
        return self._h5filename

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
        if self._readonly:
            raise ValueError("Cannot write after creating a reader")
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
        start_idx = [0, 0, 0]
        start_idx[self._slicing_dim] = start
        self._data.set_data_block((start_idx[0], start_idx[1], start_idx[2]), dataset.data)

    def _get_global_h5_filename(self) -> PathLike:
        """Creates a temporary h5 file to back the storage (using nanoseconds timestamp
        for uniqueness).
        """
        filename = str(Path(self._temppath) / f"httom_tmpstore_{time.time_ns()}.hdf5")
        filename = self.comm.bcast(filename, root=0)
        
        self._h5filename = Path(filename)
        return self._h5filename

    def _create_new_data(self, dataset: DataSet):
        try:
            data = self._create_numpy_data(self.chunk_shape, dataset.data.dtype)
            self._data = DataSet(
                data=data,
                angles=dataset.angles,
                flats=dataset.flats,
                darks=dataset.darks,
                global_shape=self._full_shape,
                global_index=self.chunk_index
            )
        except MemoryError:
            # we create a full file dataset, i.e. file-based,
            # with the full global shape in it
            # TODO: MPI processes need to sync up - if alloc fails on one, all 
            # of them need to use the file!
            # TODO: the file needs to be deleted at the end
            data = self._create_h5_data(
                self.global_shape,
                dataset.data.dtype,
                self._get_global_h5_filename(),
                self.comm,
            )
            self._data = FullFileDataSet(
                data=data,
                angles=dataset.angles,
                flats=dataset.flats,
                darks=dataset.darks,
                global_index=self.chunk_index,
                chunk_shape=self.chunk_shape,
            )
        

    @classmethod
    def _create_numpy_data(
        cls, chunk_shape: Tuple[int, int, int], dtype: DTypeLike
    ) -> np.ndarray:
        """Convenience method to enable mocking easily"""
        return np.empty(chunk_shape, dtype)

    def _create_h5_data(
        self,
        global_shape: Tuple[int, int, int],
        dtype: DTypeLike,
        file: PathLike,
        comm: MPI.Comm,
    ) -> np.ndarray:
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
    ) -> "DataSetStoreReader":
        """Create a reader from this writer, reading from the same store"""
        self._readonly = True
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
        reader = DataSetStoreReader(self, new_slicing_dim)
        # make sure finalize is called when reader object is garbage-collected
        weakref.finalize(reader, reader.finalize)
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
        self._full_shape = source.global_shape
        self._chunk_idx = source.chunk_index
        self._chunk_shape = source.chunk_shape
        if source._data is None:
            raise ValueError(
                "Cannot create DataSetStoreReader when no data has been written"
            )
            
        self._h5file: Optional[h5py.File] = None
        self._h5filename: Optional[Path] = None
        source_data = source._data
        if source.is_file_based:
            self._h5filename = source.filename
            self._h5file = h5py.File(source.filename, "r", driver="mpio", comm=self._comm)
            source_data = FullFileDataSet(
                data=self._h5file["data"],
                angles=source._data.angles,
                flats=source._data.flats,
                darks=source._data.darks,
                global_index=source.chunk_index,
                chunk_shape=source.chunk_shape,
            )

        if slicing_dim is None or slicing_dim == source.slicing_dim:
            self._slicing_dim = source.slicing_dim
            self._data = source_data
        else:
            self._data = self._reslice(source.slicing_dim, slicing_dim, source_data)
            self._slicing_dim = slicing_dim

        source.finalize()

    @property
    def is_file_based(self) -> bool:
        return self._h5filename is not None
    
    @property
    def filename(self) -> Optional[Path]:
        return self._h5filename

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

    def _reslice(
        self,
        old_slicing_dim: Literal[0, 1, 2],
        new_slicing_dim: Literal[0, 1, 2],
        data: DataSet,
    ) -> DataSet:
        assert data.is_block is False
        assert old_slicing_dim != new_slicing_dim
        if data.chunk_shape == self._full_shape:
            assert self._comm.size == 1
            return data
        else:
            assert self._comm.size > 1
            if not data.is_full:
                # we only see a chunk, so we need to do MPI-based reslicing
                array, newdim, startidx = reslice(
                    data.data, old_slicing_dim + 1, new_slicing_dim + 1, self._comm
                )
                self._chunk_shape = array.shape  #  type: ignore
                assert newdim == new_slicing_dim + 1
                idx = [0, 0, 0]
                idx[new_slicing_dim] = startidx
                self._chunk_idx = (idx[0], idx[1], idx[2])
                return  DataSet(
                    data=array,
                    angles=data.angles,
                    darks=data.darks,
                    flats=data.flats,
                    global_shape=data.global_shape,
                    global_index=self._chunk_idx,
                )
            else:
                # we have a full, file-based dataset - all we have to do 
                # is calculate the new chunk shape and start index
                rank = self._comm.rank
                nproc = self._comm.size
                length = self._full_shape[new_slicing_dim]
                startidx = round((length / nproc) * rank)
                stopidx = round((length / nproc) * (rank + 1))
                chunk_shape = list(self._full_shape)
                chunk_shape[new_slicing_dim] = stopidx-startidx
                self._chunk_shape = (chunk_shape[0], chunk_shape[1], chunk_shape[2]) 
                idx = [0, 0, 0]
                idx[new_slicing_dim] = startidx
                self._chunk_idx = (idx[0], idx[1], idx[2])
                return FullFileDataSet(
                    data=data._data,
                    angles=data.angles,
                    flats=data.flats,
                    darks=data.darks,
                    global_index=self._chunk_idx,
                    chunk_shape=self._chunk_shape
                )

    def read_block(self, start: int, length: int) -> DataSetBlock:
        block = self._data.make_block(self._slicing_dim, start, length)

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
