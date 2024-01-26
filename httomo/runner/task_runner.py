from itertools import islice
import time
from typing import Any, Dict, Literal, Optional, List, Tuple, Union
import os
from mpi4py import MPI
import httomo
import logging
from httomo.data.dataset_store import DataSetStoreWriter
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.block_split import BlockSplitter
from httomo.runner.dataset import DataSetBlock
from httomo.runner.dataset_store_interfaces import (
    DataSetSink,
    DataSetSource,
    DummySink,
    StoreBasedDataSetSink,
)
from httomo.runner.gpu_utils import get_available_gpu_memory, gpumem_cleanup
from httomo.runner.pipeline import Pipeline
from httomo.runner.platform_section import PlatformSection, sectionize
from httomo.utils import Colour, Pattern, _get_slicing_dim, log_exception, log_once


log = logging.getLogger(__name__)


class TaskRunner:
    """Handles the execution of a pipeline"""

    def __init__(
        self,
        pipeline: Pipeline,
        reslice_dir: os.PathLike,
    ):
        self.pipeline = pipeline
        self.reslice_dir = reslice_dir
        self.comm = MPI.COMM_WORLD

        self.start_time: float = 0
        self.global_stats: List = []
        self.side_outputs: Dict[str, Any] = dict()
        self.source: Optional[DataSetSource] = None
        self.sink: Optional[Union[DataSetSink, StoreBasedDataSetSink]] = None
        self.method_index: int = 1

        self.output_colour_list = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
        self.output_colour_list_short = [Colour.GREEN, Colour.CYAN]

    def execute(self) -> None:
        self.start_time = MPI.Wtime()

        sections = self._sectionize()
        
        self._prepare()
        for i, section in enumerate(sections):
            self._execute_section(section, i)
            gpumem_cleanup()
            self.method_index += len(section)
            
    def _sectionize(self) -> List[PlatformSection]:
        sections = sectionize(self.pipeline)
        num_reslices = len([s for s in sections if s.reslice]) 
        if self.pipeline.loader_reslice:
            num_reslices += 1
        if num_reslices > 1:
            log_once(
                f"WARNING: Reslicing will be performed {num_reslices} times. The number of reslices increases the total run time.",
                comm=self.comm,
                colour=Colour.RED,
            )

        return sections

    def _execute_section(self, section: PlatformSection, section_index: int = 0):
        self._setup_source_sink(section)
        assert self.source is not None, "Dataset has not been loaded yet"
        assert self.sink is not None, "Sink setup failed"

        slicing_dim_section: Literal[0, 1] = _get_slicing_dim(section.pattern) - 1  # type: ignore
        self.determine_max_slices(section, slicing_dim_section)

        self._log_pipeline(
            f"Maximum amount of slices is {section.max_slices} for section {section_index}",
            level=1,
        )

        splitter = BlockSplitter(self.source, section.max_slices)
        for block in splitter:
            self.sink.write_block(self._execute_section_block(section, block))
            gpumem_cleanup()

    def _setup_source_sink(self, section: PlatformSection):
        assert self.source is not None, "Dataset has not been loaded yet"

        slicing_dim_section: Literal[0, 1] = _get_slicing_dim(section.pattern) - 1  # type: ignore

        if self.sink is not None:
            # we have a store-based sink from the last section - use that to determine
            # the source for this one
            assert isinstance(self.sink, StoreBasedDataSetSink)
            self.source = self.sink.make_reader(slicing_dim_section)

        if section.is_last:
            # we don't need to store the results - this sink just discards it
            self.sink = DummySink(slicing_dim_section)
        else:
            self.sink = DataSetStoreWriter(
                self.source.global_shape[slicing_dim_section],
                slicing_dim_section,
                (0, 0),
                self.source.chunk_shape[slicing_dim_section],
                self.source.chunk_index[slicing_dim_section],
                self.comm,
                self.reslice_dir,
            )

    def _execute_section_block(
        self, section: PlatformSection, block: DataSetBlock
    ) -> DataSetBlock:
        for i, m in enumerate(section):
            log.debug(
                f"{m.method_name}: input shape, dtype: {block.data.shape}, {block.data.dtype}"
            )
            block = self._execute_method(m, self.method_index + i, block)
            log.debug(
                f"{m.method_name}: output shape, dtype: {block.data.shape}, {block.data.dtype}"
            )
        return block

    def _log_pipeline(self, str: str, level: int = 0, colour=Colour.BVIOLET):
        log_once(str, comm=self.comm, colour=colour, level=level)

    def _prepare(self):
        self._log_pipeline(
            f"See the full log file at: {httomo.globals.run_out_dir}/user.log",
            colour=Colour.BVIOLET,
        )
        self._check_params_for_sweep()
        self._load_datasets()

    def _load_datasets(self):
        start_time = self._log_task_start(
            self.method_index,
            self.pipeline.loader.pattern,
            self.pipeline.loader.method_name,
        )
        self.source = self.pipeline.loader.make_data_source()
        self._log_task_end(
            self.method_index,
            start_time,
            self.pipeline.loader.pattern,
            self.pipeline.loader.method_name,
            self.pipeline.loader.package_name,
        )
        self.method_index += 1

    def _execute_method(
        self, method: MethodWrapper, num: int, dataset: DataSetBlock
    ) -> DataSetBlock:
        start_time = self._log_task_start(num, method.pattern, method.method_name)
        dataset = method.execute(dataset)
        if dataset.is_last_in_chunk:
            self.update_side_inputs(method.get_side_output())
        self._log_task_end(
            num, start_time, method.pattern, method.method_name, method.package_name
        )
        return dataset

    def update_side_inputs(self, side_outputs: Dict[str, Any]):
        """Updates the methods not yet executed in the pipeline with the side outputs
        gathered so far if needed"""
        if len(side_outputs) == 0:
            return

        self.side_outputs |= side_outputs

        # iterate starting after current method index
        for m in islice(self.pipeline, self.method_index - 1, None):
            for k, v in self.side_outputs.items():
                if k in m.parameters:
                    m[k] = v

    def _log_task_start(self, num: int, pattern: Pattern, name: str) -> int:
        log_once(
            f"Running task {num} (pattern={pattern.name}): {name}...",
            self.comm,
            colour=Colour.LIGHT_BLUE,
            level=0,
        )
        return time.perf_counter_ns()

    def _log_task_end(
        self,
        num: int,
        start_time: int,
        pattern: Pattern,
        name: str,
        package: str = "httomo",
    ):
        output_str_list = [
            f"    Finished task {num} (pattern={pattern.name}): {name} (",
            package,
            f") Took {float(time.perf_counter_ns() - start_time)*1e-6:.2f}ms",
        ]
        log_once(output_str_list, comm=self.comm, colour=self.output_colour_list)

    def _check_params_for_sweep(self):
        # Check pipeline for the number of parameter sweeps present. If one is
        # defined, raise an error, due to not supporting parameter sweeps in a
        # "performance" run of httomo
        params = [m.config_params for m in self.pipeline]
        no_of_sweeps = sum(map(self._count_tuple_values, params))
        if no_of_sweeps > 0:
            err_str = (
                f"There exists {no_of_sweeps} parameter sweep(s) in the "
                "pipeline, but parameter sweeps are not supported in "
                "`httomo performance`. Please either:\n  1) Remove the parameter "
                "sweeps.\n  2) Use `httomo preview` to run this pipeline."
            )
            log_exception(err_str)
            raise ValueError(err_str)

    def _count_tuple_values(self, d: Dict[str, Any]) -> int:
        return sum(1 for v in d.values() if isinstance(v, tuple))

    def determine_max_slices(self, section: PlatformSection, slicing_dim: int):
        assert self.source is not None
        data_shape = self.source.chunk_shape

        max_slices = data_shape[slicing_dim]
        if not section.gpu:
            section.max_slices = max_slices
            return

        ###### GPU ##################

        nsl_dim_l = list(data_shape)
        nsl_dim_l.pop(slicing_dim)
        non_slice_dims_shape = (nsl_dim_l[0], nsl_dim_l[1])

        available_memory = get_available_gpu_memory(10.0)
        available_memory_in_GB = round(available_memory / (1024**3), 2)
        memory_str = (
            f"The amount of the available GPU memory is {available_memory_in_GB} GB"
        )
        log_once(memory_str, comm=self.comm, colour=Colour.BVIOLET, level=1)
        max_slices_methods = [max_slices] * len(section)

        # loop over all methods in section
        for idx, m in enumerate(section):
            if len(m.memory_gpu) == 0:
                max_slices_methods[idx] = max_slices
                continue

            output_dims = m.calculate_output_dims(non_slice_dims_shape)
            (slices_estimated, available_memory) = m.calculate_max_slices(
                self.source.dtype,
                non_slice_dims_shape,
                available_memory,
                self.source.darks,
                self.source.flats,
            )
            max_slices_methods[idx] = min(max_slices, slices_estimated)
            non_slice_dims_shape = output_dims

        section.max_slices = min(max_slices_methods)
