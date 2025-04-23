import math
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import DataSetSource


class BlockSplitter:
    """
    Produces blocks from a `DataSetSource` according to the given max slices per block.

    Notes
    -----
    A slice of the data is stored in the `DataSetBlock` that is returned, no copy is made.

    Examples
    --------
    Provides an iterator interface, so it can be used as::

         splitter = BlockSplitter(source, max_slices)
         for block in splitter:
             process_block(block)

    where `block` is a `DataSetBlock` instance.
    """

    def __init__(self, source: DataSetSource, max_slices: int):
        self._source = source
        self._chunk_size = source.chunk_shape[source.slicing_dim]
        self._max_slices = int(min(max_slices, self._chunk_size))
        self._num_blocks = math.ceil(self._chunk_size / self._max_slices)
        self._current = 0
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

    def __iter__(self):
        return self

    def __next__(self) -> DataSetBlock:
        if self._current >= len(self):
            raise StopIteration
        v = self[self._current]
        self._current += 1
        return v
