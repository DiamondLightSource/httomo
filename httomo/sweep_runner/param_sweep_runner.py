from typing import Optional

from httomo.runner.block_split import BlockSplitter
from httomo.runner.dataset import DataSetBlock
from httomo.runner.pipeline import Pipeline


class ParamSweepRunner:
    def __init__(
        self,
        pipeline: Pipeline,
    ) -> None:
        self._pipeline = pipeline
        self._block: Optional[DataSetBlock] = None

    @property
    def block(self) -> DataSetBlock:
        if self._block is None:
            raise ValueError("Block from input data has not yet been loaded")
        return self._block

    def prepare(self):
        """
        Load single block containing small number of sinogram slices in input data
        """
        source = self._pipeline.loader.make_data_source()
        splitter = BlockSplitter(source, source.global_shape[source.slicing_dim])
        self._block = splitter[0]
