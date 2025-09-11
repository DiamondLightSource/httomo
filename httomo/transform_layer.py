import os
from typing import Optional
from httomo.method_wrappers import make_method_wrapper
from httomo.runner.output_ref import OutputRef
from httomo.method_wrappers.datareducer import DatareducerWrapper
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.method_wrappers.images import ImagesWrapper
from httomo.method_wrappers.stats_calc import StatsCalcWrapper
from httomo.method_wrappers.save_intermediate import SaveIntermediateFilesWrapper
from httomo.runner.pipeline import Pipeline
from mpi4py import MPI
import httomo

from httomo_backends.methods_database.query import MethodDatabaseRepository


def _check_if_pipeline_has_a_sweep(pipeline: Pipeline) -> tuple[bool, Optional[str]]:
    pipeline_is_sweep = False
    method_to_rescale = None
    for i, m in enumerate(pipeline):
        is_last = i == len(pipeline) - 1
        if m.sweep:
            pipeline_is_sweep = True
            if is_last:
                method_to_rescale = m.method_name
        if is_last and "recon" in m.module_path:
            # reconstruction is the last method but not sweep, then we also rescale
            method_to_rescale = m.method_name
    return (pipeline_is_sweep, method_to_rescale)

class TransformLayer:
    def __init__(
        self,
        comm: MPI.Comm,
        repo=MethodDatabaseRepository(),
        save_all=False,
        out_dir: Optional[os.PathLike] = None,
    ):
        self._repo = repo
        self._save_all = save_all
        self._comm = comm
        self._out_dir = out_dir if out_dir is not None else httomo.globals.run_out_dir

    def transform(self, pipeline: Pipeline) -> Pipeline:
        pipeline = self.insert_data_reducer(pipeline)
        pipeline_is_sweep, method_to_rescale = _check_if_pipeline_has_a_sweep(pipeline)

        if pipeline_is_sweep:
            pipeline = self.insert_save_images_after_sweep(pipeline)
            pipeline = self.insert_globstats_after_sweep(pipeline, method_to_rescale)
            pipeline = self.insert_rescaletoint_after_stats_sweep(pipeline)
        else:
            pipeline = self.insert_save_methods(pipeline)

        return pipeline

    def insert_save_methods(self, pipeline: Pipeline) -> Pipeline:
        loader = pipeline.loader
        methods = []
        for m in pipeline:
            methods.append(m)
            if (
                (m.save_result or self._save_all)
                and m.method_name != "save_to_images"
                and "center" not in m.method_name
            ):
                methods.append(
                    SaveIntermediateFilesWrapper(
                        self._repo,
                        "httomo.methods",
                        "save_intermediate_data",
                        comm=self._comm,
                        save_result=False,
                        loader=loader,
                        prev_method=m,
                        task_id=f"save_{m.task_id}",
                        out_dir=self._out_dir,
                    )
                )
        return Pipeline(loader, methods)

    def insert_data_reducer(self, pipeline: Pipeline) -> Pipeline:
        """This will always insert data reducer as method 0 right after the loader"""
        loader = pipeline.loader
        methods = []
        methods.append(
            DatareducerWrapper(
                self._repo,
                "httomolib.misc.morph",
                "data_reducer",
                comm=self._comm,
                save_result=False,
                task_id="reducer_0",
            ),
        )
        for m in pipeline:
            methods.append(m)
        return Pipeline(loader, methods)

    def insert_save_images_after_sweep(self, pipeline: Pipeline) -> Pipeline:
        """For sweep methods we add image saving method after the global statistics and rescaler methods."""
        loader = pipeline.loader
        methods = []
        sweep_before = False
        for m in pipeline:
            methods.append(m)
            if m.sweep or "recon" in m.module_path and sweep_before:
                methods.append(
                    ImagesWrapper(
                        self._repo,
                        "httomolib.misc.images",
                        "save_to_images",
                        comm=self._comm,
                        save_result=False,
                        task_id=f"saveimage_sweep_{m.task_id}",
                        subfolder_name="images_sweep_" + str(m.method_name),
                        axis=1,
                    ),
                )
                sweep_before = True
        return Pipeline(loader, methods)

    def insert_globstats_after_sweep(
        self, pipeline: Pipeline, method_to_rescale: Optional[str]
    ) -> Pipeline:
        """Global statistics method is inserted to perform data rescaling before image saving"""
        methods = []
        for m in pipeline:
            methods.append(m)
            # we need to make sure that we add global stats and rescale only once
            if method_to_rescale is not None and m.method_name == method_to_rescale:
                methods.append(
                    StatsCalcWrapper(
                        self._repo,
                        "httomo.methods",
                        "calculate_stats",
                        comm=MPI.COMM_WORLD,
                        save_result=False,
                        output_mapping={"glob_stats": "glob_stats"},
                    )
                )
        return Pipeline(pipeline.loader, methods)

    def insert_rescaletoint_after_stats_sweep(self, pipeline: Pipeline) -> Pipeline:
        """Data rescaler goes after global statistics method. Note that currently the intermediate data will be also saved as rescaled."""
        methods = []
        for m in pipeline:
            methods.append(m)
            if m.method_name == "calculate_stats":
                methods.append(
                    GenericMethodWrapper(
                        self._repo,
                        "httomolib.misc.rescale",
                        "rescale_to_int",
                        comm=self._comm,
                        save_result=False,
                        perc_range_min=5.0,
                        perc_range_max=95.0,
                        bits=16,
                        glob_stats=OutputRef(
                            mapped_output_name="glob_stats",
                            method=m,
                        ),
                    )
                )
        return Pipeline(pipeline.loader, methods)
