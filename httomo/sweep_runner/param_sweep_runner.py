import os
from typing import Any, Dict, List, Optional, Tuple

import tqdm
from mpi4py.MPI import Comm

import httomo
from httomo.data.param_sweep_store import ParamSweepReader, ParamSweepWriter
from httomo.method_wrappers.images import ImagesWrapper
from httomo.runner.block_split import BlockSplitter
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.pipeline import Pipeline
from httomo.sweep_runner.param_sweep_block import ParamSweepBlock
from httomo.sweep_runner.side_output_manager import SideOutputManager
from httomo.sweep_runner.stages import NonSweepStage, Stages, SweepStage
from httomo.utils import catchtime, log_exception, log_once
from httomo.runner.gpu_utils import get_available_gpu_memory
from httomo.preview import PreviewConfig, PreviewDimConfig
from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.sweep_runner.paganin_kernel import paganin_kernel_estimator

import numpy as np


class ParamSweepRunner:
    def __init__(
        self,
        pipeline: Pipeline,
        comm: Comm,
        side_output_manager: SideOutputManager = SideOutputManager(),
    ) -> None:
        self._pipeline = pipeline
        self._vertical_slices_preview = 5
        self._side_output_manager = side_output_manager
        self._block: Optional[ParamSweepBlock] = None
        self._check_params_for_sweep()
        self._stages = self.determine_stages()
        self._start_sweep_idx, self._stop_sweep_idx = self._determine_sweep_indices(
            comm
        )
        self._sweep_values = self._stages.sweep.values[
            self._start_sweep_idx : self._stop_sweep_idx
        ]
        self._set_image_saver_params()

    @property
    def block(self) -> ParamSweepBlock:
        if self._block is None:
            raise ValueError("Block from input data has not yet been loaded")
        return self._block

    def _determine_sweep_indices(self, comm: Comm) -> Tuple[int, int]:
        no_of_sweep_vals = len(self._stages.sweep.values)
        nprocs = comm.size
        rank = comm.rank
        start = round((no_of_sweep_vals / nprocs) * rank)
        end = round((no_of_sweep_vals / nprocs) * (rank + 1))
        return start, end

    def _check_params_for_sweep(self):
        """
        Check pipeline for the number of parameter sweeps present.

        If none are defined, then raise an error. If more than one is defined, then also raise
        an error, due to not supporting parameter sweeps over more than one parameter at a
        time.
        """
        params = [m.config_params for m in self._pipeline]
        no_of_sweeps = sum(map(_count_tuple_values, params))

        if no_of_sweeps == 0:
            err_str = "No parameter sweep detected in pipeline"
            log_exception(err_str)
            raise ValueError(err_str)

        if no_of_sweeps > 1:
            err_str = (
                "Parameter sweep over more than one parameter detected in pipeline; "
                "a sweep over only one parameter at a time is supported"
            )
            log_exception(err_str)
            raise ValueError(err_str)

    def _set_image_saver_params(self) -> None:
        """
        Set the `offset` and `watermark_vals` parameters for any `ImagesWrapper`
        instances in the pipeline.
        """
        non_sweep_stages = [self._stages.before_sweep, self._stages.after_sweep]
        for non_sweep_stage in non_sweep_stages:
            for method in non_sweep_stage.methods:
                if isinstance(method, ImagesWrapper):
                    method["offset"] = self._start_sweep_idx
                    method["watermark_vals"] = self._sweep_values

    def determine_stages(self) -> Stages:
        """
        Groups methods into "before sweep", "sweep", and "after sweep" stages
        """
        before_sweep: List[MethodWrapper] = []
        sweep_wrapper: Optional[MethodWrapper] = None
        sweep_param_name: Optional[str] = None
        sweep_param_vals: Optional[Tuple[Any, ...]] = None
        after_sweep: List[MethodWrapper] = []

        non_sweep_stage_methods: List[MethodWrapper] = before_sweep
        for method in self._pipeline:
            params = method.config_params
            no_of_sweeps = sum(map(_count_tuple_values, [params]))
            if no_of_sweeps == 0:
                non_sweep_stage_methods.append(method)
            else:
                sweep_wrapper = method
                sweep_param_name, sweep_param_vals = _get_param_sweep_name_and_vals(
                    params
                )
                non_sweep_stage_methods = after_sweep

        assert sweep_wrapper is not None
        assert sweep_param_name is not None
        assert sweep_param_vals is not None

        return Stages(
            before_sweep=NonSweepStage(before_sweep),
            sweep=SweepStage(
                method=sweep_wrapper,
                param_name=sweep_param_name,
                values=sweep_param_vals,
            ),
            after_sweep=NonSweepStage(after_sweep),
        )

    def modify_loader_preview(self):
        """
        Modifies loader's preview based on the method present in the pipeline.
        In some cases we need to override users given preview, e.g., when Paganin filter is present.
        """

        # we call this only to find the global size of the data (not very efficient)
        source = self._pipeline.loader.make_data_source(padding=(0, 0))

        if source.raw_shape[1] < self._vertical_slices_preview:
            err_str = (
                "The size of the raw data  "
                f"{source.raw_shape[1]} is smaller than it is required by the sweep run - "
                f"{self._vertical_slices_preview}. "
                f"Please consider using a larger input dataset with the sweep run."
            )
            raise ValueError(err_str)

        # before modifying preview here we need to check if the block fits the memory if Paganin method (TomoPy implementation) is present in the pipeline
        for method in self._pipeline._methods:
            if "paganin_filter_tomopy" in method.method_name and method.sweep:
                # Specifically dealing with the Paganin filter variable kernel size, as if the kernel is large,
                # the larger preview needs to be taken in that case. So far this is the only method that
                # requires an extended preview.

                vertical_slices_preview_Paganin = paganin_kernel_estimator(
                    method.config_params["pixel_size"],
                    method.config_params["alpha"],
                    method.config_params["energy"],
                    method.config_params["dist"],
                    vert_min_limit=self._vertical_slices_preview,
                    peak_height=0.01,
                )
                self._vertical_slices_preview = np.max(vertical_slices_preview_Paganin)

                vertical_slices_preview_max = _slices_to_fit_memory_Paganin(source)

                # we do not stop the run if the Paganin's kernel is too wide, but extend the preview to the maximum allowed size for the current card
                if self._vertical_slices_preview > vertical_slices_preview_max:
                    self._vertical_slices_preview = vertical_slices_preview_max

        preview_new_start_stop = _preview_modifier(
            self._pipeline.loader.preview,
            source.raw_shape[1],
            self._vertical_slices_preview,
        )

        self._pipeline.loader.preview = PreviewConfig(
            angles=self._pipeline.loader.preview.angles,
            detector_y=PreviewDimConfig(
                start=preview_new_start_stop[0], stop=preview_new_start_stop[1]
            ),
            detector_x=self._pipeline.loader.preview.detector_x,
        )

    def prepare(self):
        """
        Load single block containing small number of sinogram slices in input data
        """
        source = self._pipeline.loader.make_data_source(padding=(0, 0))

        SINO_SLICING_DIM = 1
        log_once(f"Loading data with shape {source.global_shape}")
        splitter = BlockSplitter(source, source.global_shape[source.slicing_dim])
        dataset_block = splitter[0]
        sweep_block = ParamSweepBlock(
            data=dataset_block.data,
            aux_data=dataset_block.aux_data,
            slicing_dim=SINO_SLICING_DIM,
        )
        self._block = sweep_block

    def _execute_non_sweep_stage(self, stage: NonSweepStage):
        assert self._block is not None
        for method in stage.methods:
            log_once(f"Running {method.method_name} ({method.package_name})")
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
        writer = ParamSweepWriter(len(self._sweep_values))
        method = self._stages.sweep.method

        log_once(f"Running {method.method_name} ({method.package_name})")
        log_once("    Beginning parameter sweep")
        log_once(f"    Parameter name: {self._stages.sweep.param_name}")
        total_vals_str = (
            "    Total number of values across all processes: "
            f"{len(self._stages.sweep.values)}"
        )
        log_once(total_vals_str)
        log_once(f"    Values executed in this process: {len(self._sweep_values)}")

        # Redirect tqdm progress bar output to /dev/null, and instead manually write sweep
        # progress to logfile within loop
        progress = tqdm.tqdm(
            iterable=self._sweep_values,
            file=open(os.devnull, "w"),
            unit="value",
            ascii=True,
        )
        for val, _ in zip(self._sweep_values, progress):
            # Blocks are modified in-place by method wrappers, so a new block must be created
            # that contains a copy of the input data to the sweep stage
            block = ParamSweepBlock(
                data=self.block.data.copy(),
                aux_data=self.block.aux_data,
                slicing_dim=self.block.slicing_dim,
            )
            self._side_output_manager.update_params(method)
            method.append_config_params({self._stages.sweep.param_name: val})
            progress.set_postfix_str(f"{self._stages.sweep.param_name}={val}")
            log_once(f"    {str(progress)}")
            block = method.execute(block)
            if len(method.get_side_output().keys()) > 0:
                raise ValueError(
                    "Producing a side output is not supported in parameter sweep methods"
                )
            writer.write_sweep_result(block)

        log_once("    Finished parameter sweep")

        reader = ParamSweepReader(writer)
        self._block = reader.read_sweep_results()

    def execute(self):
        """Load input data and execute all stages (before sweep, sweep, after sweep)"""
        with catchtime() as t:
            log_once(f"See the full log file at: {httomo.globals.run_out_dir}/user.log")
            self.modify_loader_preview()
            self.prepare()
            self.execute_before_sweep()
            self.execute_sweep()
            self.execute_after_sweep()
        log_once(f"Pipeline finished. Took {t.elapsed:.3f}s")


def _count_tuple_values(d: Dict[str, Any]) -> int:
    return sum(1 for v in d.values() if isinstance(v, tuple))


def _get_param_sweep_name_and_vals(d: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
    param_name: Optional[str] = None
    sweep_vals: Optional[Tuple[Any, ...]] = None

    for k, v in d.items():
        if isinstance(v, tuple):
            param_name = k
            sweep_vals = v

    assert param_name is not None
    assert sweep_vals is not None
    return param_name, sweep_vals


def _preview_modifier(
    preview_old: PreviewConfig, det_y_dim_raw: int, vert_slices: int
) -> List[int]:
    """
    Modifies the size of the user-defined preview (expand it or shrink it)
    """

    preview_new = [0, det_y_dim_raw]
    range_preview = preview_old.detector_y.stop - preview_old.detector_y.start

    start_new = preview_old.detector_y.start + range_preview // 2 - vert_slices // 2
    stop_new = preview_old.detector_y.stop - range_preview // 2 + vert_slices // 2
    if start_new >= 0:
        preview_new[0] = start_new
    if stop_new < det_y_dim_raw:
        preview_new[1] = stop_new
    return preview_new


def _slices_to_fit_memory_Paganin(source: DataSetSource) -> int:
    """
    Estimating the number of vertical slices than can fit the device to run Paganin method
    """
    factor = 10  # ad-hoc parameter as no real memory estimation is performed. Should be increased if OOM happens.
    available_memory = get_available_gpu_memory(10.0)
    available_memory_in_GB = round(available_memory / (1024**3), 2)
    tot_slices_fit = int(
        available_memory_in_GB
        // (((source.aux_data.angles_length * source.global_shape[2]) * 4) / 1024**3)
    )
    return tot_slices_fit // factor
