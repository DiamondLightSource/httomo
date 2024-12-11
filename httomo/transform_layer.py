import os
from pathlib import Path
from typing import Optional
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.datareducer import DatareducerWrapper
from httomo.method_wrappers.generic import GenericMethodWrapper
from httomo.method_wrappers.images import ImagesWrapper
from httomo.method_wrappers.save_intermediate import SaveIntermediateFilesWrapper
from httomo.runner.pipeline import Pipeline
from mpi4py import MPI
import httomo

from httomo_backends.methods_database.query import MethodDatabaseRepository


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
        pipeline = self.insert_save_methods(pipeline)
        pipeline = self.insert_data_reducer(pipeline)
        pipeline = self.insert_save_images_after_sweep(
            pipeline
        )  # will be applied to sweep methods only
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
        """For sweep methods we add image saving method after, and also a rescaling method to
        rescale the data passed to the image saver. In addition we also add saving the results
        of the reconstruction, if the module is present"""
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
