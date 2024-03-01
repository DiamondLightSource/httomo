from itertools import islice
import time
from typing import Any, Dict, Literal, Optional, List, Union
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
    ReadableDataSetSink,
)
from httomo.runner.gpu_utils import get_available_gpu_memory, gpumem_cleanup
from httomo.runner.pipeline import Pipeline
from httomo.runner.section import Section, sectionize
from httomo.utils import Colour, Pattern, _get_slicing_dim, log_exception, log_once
import numpy as np

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
        self.sink: Optional[Union[DataSetSink, ReadableDataSetSink]] = None

        self.output_colour_list = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
        self.output_colour_list_short = [Colour.GREEN, Colour.CYAN]

    def execute(self) -> None:
        self.start_time = MPI.Wtime()

        sections = self._sectionize()
        
        self._prepare()
        for i, section in enumerate(sections):
            self._execute_section(section, i)
            gpumem_cleanup()
            
        self.end_time = MPI.Wtime()
        self._log_pipeline(f"Pipeline finished. Took {self.end_time-self.start_time:.3f}s")
            
    def _sectionize(self) -> List[Section]:
        sections = sectionize(self.pipeline)
        self._log_pipeline(f"Pipeline has been separated into {len(sections)} sections")
        num_stores = len(sections) - 1 
        if num_stores > 1:
            log_once(
                f"WARNING: Reslicing will be performed {num_stores} times. The number of reslices increases the total run time.",
                comm=self.comm,
                colour=Colour.RED,
            )

        return sections

    def _execute_section(self, section: Section, section_index: int = 0):
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

    def _setup_source_sink(self, section: Section):
        assert self.source is not None, "Dataset has not been loaded yet"

        slicing_dim_section: Literal[0, 1] = _get_slicing_dim(section.pattern) - 1  # type: ignore

        if self.sink is not None:
            # we have a store-based sink from the last section - use that to determine
            # the source for this one
            assert isinstance(self.sink, ReadableDataSetSink)
            self.source = self.sink.make_reader(slicing_dim_section)

        if section.is_last:
            # we don't need to store the results - this sink just discards it
            self.sink = DummySink(slicing_dim_section)
        else:
            self.sink = DataSetStoreWriter(
                slicing_dim_section,
                self.comm,
                self.reslice_dir,
            )

    def _execute_section_block(
        self, section: Section, block: DataSetBlock
    ) -> DataSetBlock:
        for method in section:
            self.set_side_inputs(method)
            block = self._execute_method(method, block)
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
            "loader",
            self.pipeline.loader.pattern,
            self.pipeline.loader.method_name,
        )
        self.source = self.pipeline.loader.make_data_source()
        self._log_task_end(
            "loader",
            start_time,
            self.pipeline.loader.pattern,
            self.pipeline.loader.method_name,
            self.pipeline.loader.package_name,
        )

    def _execute_method(
        self, method: MethodWrapper, block: DataSetBlock
    ) -> DataSetBlock:
        start_time = self._log_task_start(method.task_id, method.pattern, method.method_name)
        block = method.execute(block)
        if block.is_last_in_chunk:
            self.append_side_outputs(method.get_side_output())
        self._log_task_end(
            method.task_id, start_time, method.pattern, method.method_name, method.package_name
        )
        return block

    def append_side_outputs(self, side_outputs: Dict[str, Any]):
        """Appends to the side outputs that are available for the next section(s)"""
        if len(side_outputs) == 0:
            return

        self.side_outputs |= side_outputs

    def set_side_inputs(self, method: MethodWrapper):
        """Sets the parameters that reference side outputs for this method."""
        for k, v in self.side_outputs.items():
            if k in method.parameters:
                method[k] = v

    def _log_task_start(self, id: str, pattern: Pattern, name: str) -> int:
        log_once(
            f"Running {id} (pattern={pattern.name}): {name}...",
            self.comm,
            colour=Colour.LIGHT_BLUE,
            level=0,
        )
        return time.perf_counter_ns()

    def _log_task_end(
        self,
        id: str,
        start_time: int,
        pattern: Pattern,
        name: str,
        package: str = "httomo",
    ):
        output_str_list = [
            f"    Finished {id} (pattern={pattern.name}): {name} (",
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

    def determine_max_slices(self, section: Section, slicing_dim: int):
        assert self.source is not None
        data_shape = self.source.chunk_shape

        max_slices = data_shape[slicing_dim]
        if len(section) == 0:
            section.max_slices = min(httomo.globals.MAX_CPU_SLICES, max_slices)
            return

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
        has_gpu = False
        for idx, m in enumerate(section):
            if len(m.memory_gpu) == 0:
                max_slices_methods[idx] = max_slices
                continue

            has_gpu = has_gpu or m.is_gpu
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

        if not has_gpu:
            section.max_slices = min(min(max_slices_methods), httomo.globals.MAX_CPU_SLICES)
        else:
            section.max_slices = min(max_slices_methods)
