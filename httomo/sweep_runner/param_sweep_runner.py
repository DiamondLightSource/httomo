from typing import List, Optional

from httomo.data.param_sweep_store import ParamSweepReader, ParamSweepWriter
from httomo.runner.block_split import BlockSplitter
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.pipeline import Pipeline
from httomo.sweep_runner.param_sweep_block import ParamSweepBlock
from httomo.sweep_runner.side_output_manager import SideOutputManager
from httomo.sweep_runner.stages import Stages


class ParamSweepRunner:
    def __init__(
        self,
        pipeline: Pipeline,
        stages: Stages,
        side_output_manager: SideOutputManager = SideOutputManager(),
    ) -> None:
        self._sino_slices_threshold = 7
        self._pipeline = pipeline
        self._stages = stages
        self._side_output_manager = side_output_manager
        self._block: Optional[ParamSweepBlock] = None

    @property
    def block(self) -> ParamSweepBlock:
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
        dataset_block = splitter[0]
        sweep_block = ParamSweepBlock(
            data=dataset_block.data,
            aux_data=dataset_block.aux_data,
        )
        self._block = sweep_block

    def _execute_non_sweep_stage(self, wrappers: List[MethodWrapper]):
        assert self._block is not None
        for method in wrappers:
            self._side_output_manager.update_params(method)
            self._block = method.execute(self._block)
            self._side_output_manager.append(method.get_side_output())

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
            block = ParamSweepBlock(
                data=self.block.data.copy(),
                aux_data=self.block.aux_data,
            )
            self._side_output_manager.update_params(method)
            block = method.execute(block)
            if len(method.get_side_output().keys()) > 0:
                raise ValueError(
                    "Producing a side output is not supported in parameter sweep methods"
                )
            writer.write_sweep_result(block)

        reader = ParamSweepReader(writer)
        self._block = reader.read_sweep_results()

    def execute(self):
        """Load input data and execute all stages (before sweep, sweep, after sweep)"""
        self.prepare()
        self.execute_before_sweep()
        self.execute_sweep()
        self.execute_after_sweep()
