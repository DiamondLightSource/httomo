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


def _check_if_pipeline_has_a_sweep(pipeline: Pipeline) -> bool:
    pipeline_is_sweep = False
    for i, m in enumerate(pipeline):
        if m.sweep:
            pipeline_is_sweep = True
    return pipeline_is_sweep


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
        pipeline_is_sweep = _check_if_pipeline_has_a_sweep(pipeline)

        pipeline = self.insert_data_reducer(pipeline)
        if pipeline_is_sweep:
            pipeline = self.remove_redundant_method_in_sweep(pipeline)
        pipeline = self.insert_data_checker(pipeline)

        if pipeline_is_sweep:
            pipeline = self.insert_save_images_after_sweep(pipeline)
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

    def insert_data_checker(self, pipeline: Pipeline) -> Pipeline:
        """This will insert CPU or GPU data checker method AFTER most of the methods in the pipeline"""
        loader = pipeline.loader
        methods = []
        for index, m in enumerate(pipeline):
            methods.append(m)
            # handling some exceptions here after which we don't need to insert the data checker
            exceptions_methods = [
                "data_reducer",
                "data_checker",
                "calculate_stats",
                "rescale_to_int",
                "save_to_images",
            ]
            if (
                m.method_name not in exceptions_methods
                and "rotation" not in m.module_path
                and index < len(pipeline._methods) - 1
            ):
                if m.is_cpu:
                    # add the CPU checker method
                    methods.append(
                        GenericMethodWrapper(
                            self._repo,
                            "httomolib.misc.utils",
                            "data_checker",
                            comm=self._comm,
                            save_result=False,
                            task_id=f"datachecker_{m.task_id}",
                            infsnans_correct=True,
                            zeros_warning=True,
                            data_to_method_name=m.method_name,
                        ),
                    )
                else:
                    # add the GPU checker method
                    methods.append(
                        GenericMethodWrapper(
                            self._repo,
                            "httomolibgpu.misc.utils",
                            "data_checker",
                            comm=self._comm,
                            save_result=False,
                            task_id=f"datachecker_{m.task_id}",
                            infsnans_correct=True,
                            zeros_warning=False,
                            data_to_method_name=m.method_name,
                        ),
                    )
        return Pipeline(loader, methods)

    def insert_save_images_after_sweep(self, pipeline: Pipeline) -> Pipeline:
        """For sweep methods we add image saving method."""
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

    def remove_redundant_method_in_sweep(self, pipeline: Pipeline) -> Pipeline:
        """Remove "redundant" methods in the sweep pipeline that were inserted by the user."""
        redundant_methods = ["calculate_stats", "rescale_to_int", "save_to_images"]
        loader = pipeline.loader
        methods = []
        for m in pipeline:
            if m.method_name not in redundant_methods:
                methods.append(m)
        return Pipeline(loader, methods)
