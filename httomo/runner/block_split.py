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
        self._chunk_size = (
            source.chunk_shape[source.slicing_dim]
            + source.padding[0]
            + source.padding[1]
        )
        self._max_slices = int(min(max_slices, self._chunk_size))

        core_slices = self._max_slices - source.padding[0] - source.padding[1]
        step = core_slices
        num_blocks = 0
        start_of_next_block = 0
        while (
            start_of_next_block + source.padding[0] + core_slices + source.padding[1]
            < self._chunk_size
        ):
            num_blocks += 1
            start_of_next_block += step

        if (
            start_of_next_block + source.padding[0] + core_slices + source.padding[1]
            >= self._chunk_size
        ):
            num_blocks += 1

        self._num_blocks = num_blocks
        self._current = 0
        assert self._source.slicing_dim in [
            0,
            1,
        ], "Only supporting slicing in projection and sinogram dimension"

    @property
    def slices_per_block(self) -> int:
        return self._max_slices - self._source.padding[0] - self._source.padding[1]

    def __len__(self):
        return self._num_blocks

    def __getitem__(self, idx: int) -> DataSetBlock:
        start = idx * self.slices_per_block
        if (
            start
            >= self._chunk_size - self._source.padding[0] - self._source.padding[1]
        ):
            raise IndexError("Index out of bounds")
        len = min(
            self.slices_per_block,
            self._chunk_size
            - self._source.padding[0]
            - self._source.padding[1]
            - start,
        )
        return self._source.read_block(start, len)
