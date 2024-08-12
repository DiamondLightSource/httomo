import math
from typing import Iterator
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import DataSetSource
import logging

log = logging.getLogger(__name__)


class BlockSplitter:
    """Produces blocks from a DataSetSource according to the given max slices
    per block. It provides an iterator interface, so that it can be used as::

         splitter = BlockSplitter(source, max_slices)
         for block in splitter:
             process_block(block)

    Where a block is a DataSet instance.

    Note that a slice of the data is returned and no copy is made.
    """

    def __init__(self, source: DataSetSource, max_slices: int):
        self._source = source
        self._chunk_size = source.chunk_shape[source.slicing_dim]
        self._max_slices = int(min(max_slices, self._chunk_size))
        self._num_blocks = math.ceil(self._chunk_size / self._max_slices)
        assert self._source.slicing_dim in [
            0,
            1,
        ], "Only supporting slicing in projection and sinogram dimension"

    @property
    def slices_per_block(self) -> int:
        return self._max_slices

    def __len__(self):
        return self._num_blocks

    def __getitem__(self, idx: int) -> DataSetBlock:
        start = idx * self.slices_per_block
        if start >= self._chunk_size:
            raise IndexError("Index out of bounds")
        len = min(self.slices_per_block, self._chunk_size - start)
        return self._source.read_block(start, len)

    def __iter__(self) -> Iterator[DataSetBlock]:
        class BlockIterator:
            def __init__(self, splitter):
                self.splitter = splitter
                self._current = 0

            def __iter__(self) -> "BlockIterator":
                return self  # pragma: no cover

            def __next__(self) -> DataSetBlock:
                if self._current >= len(self.splitter):
                    raise StopIteration
                v = self.splitter[self._current]
                self._current += 1
                return v

        return BlockIterator(self)
