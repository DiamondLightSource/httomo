from typing import List, Optional

from httomo.data.param_sweep_store import ParamSweepReader, ParamSweepWriter
from httomo.runner.block_split import BlockSplitter
from httomo.runner.dataset import DataSetBlock
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.pipeline import Pipeline
from httomo.sweep_runner.stages import Stages


class ParamSweepRunner:
    def __init__(
        self,
        pipeline: Pipeline,
        stages: Stages,
    ) -> None:
        self._sino_slices_threshold = 7
        self._pipeline = pipeline
        self._stages = stages
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

        SINO_SLICING_DIM = 1
        no_of_middle_slices = source.global_shape[SINO_SLICING_DIM]
        if no_of_middle_slices > self._sino_slices_threshold:
            err_str = (
                "Parameter sweep runs support input data containing "
                f"<= {self._sino_slices_threshold} sinogram slices, input data "
                f"contains {no_of_middle_slices} slices"
            )
            raise ValueError(err_str)

        splitter = BlockSplitter(source, source.global_shape[source.slicing_dim])
        self._block = splitter[0]

    def _execute_non_sweep_stage(self, wrappers: List[MethodWrapper]):
        assert self._block is not None
        for method in wrappers:
            self._block = method.execute(self._block)

    def execute_before_sweep(self):
        """Execute all methods before the parameter sweep"""
        self._execute_non_sweep_stage(self._stages.before_sweep)

    def execute_after_sweep(self):
        """Execute all methods after the parameter sweep"""
        self._execute_non_sweep_stage(self._stages.after_sweep)

    def execute_sweep(self):
        """Execute all param variations of the same method in the sweep"""
        writer = ParamSweepWriter(
            no_of_sweeps=len(self._stages.sweep),
            single_shape=self.block.global_shape,
        )

        for method in self._stages.sweep:
            # Blocks are modified in-place by method wrappers, so a new block must be created
            # that contains a copy of the input data to the sweep stage
            block = DataSetBlock(
                data=self.block.data.copy(),
                aux_data=self.block.aux_data,
                slicing_dim=self.block.slicing_dim,
                global_shape=self.block.global_shape,
                chunk_start=0,
                chunk_shape=self.block.chunk_shape,
                block_start=0,
            )
            block = method.execute(block)
            writer.write_sweep_result(block)

        reader = ParamSweepReader(writer)
        self._block = reader.read_sweep_results()
