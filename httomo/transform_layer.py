import os
from pathlib import Path
from typing import Optional
from httomo.method_wrappers import make_method_wrapper
from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.pipeline import Pipeline
from mpi4py import MPI
import httomo

class TransformLayer:
    def __init__(
        self,
        repo=MethodDatabaseRepository(),
        comm: MPI.Comm = MPI.COMM_WORLD,
        save_all=False,
        out_dir: Optional[os.PathLike] = None
    ):
        self._repo = repo
        self._save_all = save_all
        self._comm = comm
        self._out_dir = out_dir if out_dir is not None else httomo.globals.run_out_dir

    def transform(self, pipeline: Pipeline) -> Pipeline:
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
                    make_method_wrapper(
                        self._repo,
                        "httomo.methods",
                        "save_intermediate_data",
                        comm=self._comm,
                        save_result=False,
                        loader=loader,
                        prev_method=m,
                        task_id=f"save_{m.task_id}",
                        out_dir=self._out_dir
                    )
                )
        return Pipeline(loader, methods)
