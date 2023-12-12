import yaml
from pathlib import Path
from typing import Any, Dict, List, Protocol, TypeAlias
from importlib import import_module, util
import sys
import os

from mpi4py import MPI
from mpi4py.MPI import Comm

import httomo
from httomo.logger import setup_logger

from httomo.utils import log_exception
from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.pipeline import Pipeline

from httomo.runner.backend_wrapper import make_backend_wrapper
from httomo.runner.loader import make_loader
from httomo.runner.output_ref import OutputRef
from httomo.runner.task_runner import TaskRunner

from datetime import datetime

MethodConfig: TypeAlias = Dict[str, Dict[str, Any]]
PipelineStageConfig: TypeAlias = List[MethodConfig]
PipelineConfig: TypeAlias = List[PipelineStageConfig]

class UiLayer:
    """A common user interface for different front-ends in httomo.
    We currently support YAML and Python interfaces, but in future
    different UI's can be added based on other data formats, e.g. JSON. 
    """
    
    def __init__(
        self,
        tasks_file_path: str,
        in_data_file_path: str,
        comm: Comm,
    ):   
        
        self.repo = MethodDatabaseRepository()
        self.tasks_file_path = tasks_file_path
        self.in_data_file = in_data_file_path
        self.comm = comm
        
        root, ext = os.path.splitext(self.tasks_file_path)
        if ext == '.yaml':
            # loading yaml file with tasks provided
            self.PipelineStageConfig = _yaml_loader(self.tasks_file_path)[0]
        elif ext == '.py':
            # loading python file with tasks provided
            self.PipelineStageConfig = _python_tasks_loader(self.tasks_file_path)
        else:
            raise ValueError(
                f"The extension {ext} of the file {root} with tasks is unknown."
            )

    def build_pipeline(self) -> Pipeline:       
        output_mapping = {} # TODO
        methods_list = []
        for task_conf in self.PipelineStageConfig:
            if "loaders" in task_conf['module_path']:
                task_conf['parameters']['in_file'] = self.in_data_file
                # unpack params and initiate a loader
                loader =  make_loader(
                        self.repo,
                        task_conf['module_path'],
                        task_conf['method'],
                        self.comm,
                        **task_conf['parameters'],
                        )
            else:
                if "parameters" not in task_conf:
                    task_conf['parameters'] = {}
                # unpack params of a method and append to a list of methods
                method = make_backend_wrapper(
                    self.repo,
                    task_conf['module_path'],
                    task_conf['method'],
                    self.comm,
                    output_mapping,
                    **task_conf['parameters'],
                )
                methods_list.append(method)
        return Pipeline(
            loader=loader,
            methods=methods_list,
            main_pipeline_start=1,
        )

def _yaml_loader(file_path: str) -> list:
    with open(file_path, "r") as f:
        tasks_list = list(yaml.load_all(f, Loader=yaml.FullLoader))
    return tasks_list

def _python_tasks_loader(file_path: str) -> list:
    module_spec = util.spec_from_file_location("methods_to_list", file_path)
    foo = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(foo)
    tasks_list = list(foo.methods_to_list())
    return tasks_list

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <script> <yaml_file> <data_file>")
        exit(1)

    out_dir = Path("httomo_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    httomo.globals.run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        # Setup global logger object
        httomo.globals.logger = setup_logger(httomo.globals.run_out_dir)

    from httomo.ui_layer import *
    
    initYaml = UiLayer(sys.argv[1], sys.argv[2], comm=comm)

    pipeline = initYaml.build_pipeline()

    #print("done pipeline")
    runner = TaskRunner(pipeline, False)
    runner.execute()