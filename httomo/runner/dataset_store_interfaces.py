import abc
from typing import Literal, Optional, Protocol, Tuple
import numpy as np

from httomo.runner.dataset import DataSetBlock
from httomo.runner.auxiliary_data import AuxiliaryData

# Interfaces for an MPI-aware store for full datasets, where each process handles a *chunk*, and
# the data can be read or written in *blocks*, sliced in the slicing dimension.
#
# The source is used as an interface for readers, and sink for writers.
# Note that any implementation might implement both interfaces.
#
# Possible implementation (other approaches are thinkable):
# - Reading (DataSetSource):
#     - MPI is used to split the data into chunks for each process, which are attempted
#       to be read into memory in full in one go (assuming file source).
#     - If the available physical memory is not enough, it tries to read smaller parts
#       until it works. That part must be bigger or equal to the block size.
#     - The loader can be wrapped or implement this interface as well
#
# - Writing:
#     - data for the full chunk is allocated when the first block is written (so size in the
#       non-sliced dimensions are known, as well as data type)
#     - the number of slices per chunk has to be set/known somehow before
#     - it tries to allocate the full chunk in-memory
#     - if the allocation fails, it tries to reduce the slice and go through a file
#       for swapping data in/out of memory
#
# - Reslicing:
#     - for the loader interface, the slicing dim is simply set before the first read
#       and the data is read in that slicing direction
#     - for an implementation that is both a sink and a source,
#       after writing of the full chunk is done (or when a file is the basis),
#       `store.slicing_dim = 2` can be set to trigger a potential reslice
#     - if data is all file-based, nothing really changes - blocks will be read in
#       the other dimension with no issue
#     - if data chunk is in memory, an MPI reslice operation is triggered to update
#       the chunk

# Notes for implementation:
# - the BlockSplitter / BlockAggregator will read/write into these interfaces
# - after Aggregator has written to it, splitter can read from it in the next section
# - the loader wrapper can implement the source interface
# - perhaps the method wrapper for saving intermediate files could use an instance of
#   the sink interface - with some setting to definitely write to files.
#   finalize would take care of making sure that everything is written to disk in the end


class DataSetSource(Protocol):
    """MPI-aware source for full datasets, where each process handles a *chunk*, and
    the data can be read in *blocks*, sliced in the given slicing dimension"""

    @property
    def dtype(self) -> np.dtype: ...  # pragma: no cover

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        """Global data shape across all processes that we eventually have to read."""
        ...  # pragma: no cover

    @property
    def raw_shape(self) -> Tuple[int, int, int]:
        """Returns raw data shape"""
        ...  # pragma: no cover

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of a chunk, i.e. the data processed in the current
        MPI process (whether it fits in memory or not)"""
        ...  # pragma: no cover

    @property
    def global_index(self) -> Tuple[int, int, int]:
        """Returns the start index of the chunk within the global data array"""
        ...  # pragma: no cover

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        """Slicing dimension - 0, 1, or 2"""
        ...  # pragma: no cover

    @property
    def aux_data(self) -> AuxiliaryData:
        """Auxiliary data"""
        ...  # pragma: no cover

    def read_block(self, start: int, length: int) -> DataSetBlock:
        """Reads a block from the dataset, starting at `start` of length `length`,
        in the current slicing dimension. Note that `start` is chunk-based,
        i.e. mean different things in different processes."""
        ...  # pragma: no cover

    def finalize(self):
        """Method intended to be called after reading all blocks is done,
        to give implementations a chance to close files, free memory, etc."""
        ...  # pragma: no cover


class DataSetSink(Protocol):
    @property
    def global_shape(self) -> Tuple[int, int, int]:
        """Global data shape across all processes that we eventually have to write."""
        ...  # pragma: no cover

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of a chunk, i.e. the data processed in the current
        MPI process (whether it fits in memory or not)"""
        ...  # pragma: no cover

    @property
    def global_index(self) -> Tuple[int, int, int]:
        """Returns the start index of the chunk within the global data array"""
        ...  # pragma: no cover

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        """Slicing dimension - 0, 1, or 2"""
        ...  # pragma: no cover

    def write_block(self, block: DataSetBlock):
        """Writes a block to the store, starting at the index in dataset.chunk_index,
        in the current slicing dimension.

        NOTE: Implementers should make sure to move the dataset to CPU if required -
        it may be on GPU when this method is called."""
        ...  # pragma: no cover

    def finalize(self):
        """Method intended to be called after writing all blocks is done,
        to give implementations a chance to write everything to disk and close the file,
        etc."""
        ...  # pragma: no cover


class ReadableDataSetSink(DataSetSink):
    """Interface for a DataSetSink that is store-based, i.e. where it's possible to construct
    a reader from the same data store that this sink has been written first"""

    @abc.abstractmethod
    def make_reader(
        self, new_slicing_dim: Optional[Literal[0, 1, 2]] = None
    ) -> DataSetSource:
        """Method to make a source from this sink, which will read the data that was written,
        possibly in a new slicing dimension"""
        ...  # pragma: no cover


class DummySink(DataSetSink):
    """DataSetSink that does nothing - it purely drops the data"""

    def __init__(self, slicing_dim: Literal[0, 1, 2]):
        self._global_shape: Tuple[int, int, int] = (0, 0, 0)
        self._chunk_index: Tuple[int, int, int] = (0, 0, 0)
        self._chunk_shape: Tuple[int, int, int] = (0, 0, 0)
        self._slicing_dim = slicing_dim

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        return self._global_shape

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        return self._chunk_shape

    @property
    def global_index(self) -> Tuple[int, int, int]:
        return self._chunk_index

    @property
    def slicing_dim(self) -> Literal[0, 1, 2]:
        return self._slicing_dim

    def write_block(self, block: DataSetBlock):
        self._global_shape = block.global_shape
        self._chunk_shape = block.chunk_shape
        self._chunk_index = block.chunk_index

    def finalize(self):
        pass  # pragma: no cover
