import yaml
from pathlib import Path
from typing import Any, Dict, List, Protocol, TypeAlias
from importlib import import_module
import sys

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

from datetime import datetime


MethodConfig: TypeAlias = Dict[str, Dict[str, Any]]
PipelineStageConfig: TypeAlias = List[MethodConfig]
PipelineConfig: TypeAlias = List[PipelineStageConfig]

class YamlInterface:
    
    def __init__(
        self,
        yaml_file_tasks: str,
        in_data_file: str,
        comm: Comm,
    ):   
        
        self.repo = MethodDatabaseRepository()
        self.yaml_file_tasks = yaml_file_tasks
        self.in_data_file = in_data_file        
        self.comm = comm

    def yaml_to_pipeline(self) -> Pipeline:
        with open(self.yaml_file_tasks, "r") as f:
            yaml_conf = list(yaml.load_all(f, Loader=yaml.FullLoader))
        PipelineStageConfig = yaml_conf[0]
        output_mapping = {} # TODO

        methods_list = []
        for task_conf in PipelineStageConfig:
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
            main_pipeline_start=3,
        )

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

    from httomo.yaml_tools.yaml_to_pipeline import YamlInterface
    
    initialiseY = YamlInterface(sys.argv[1], sys.argv[2], comm=comm)

    pipeline = initialiseY.yaml_to_pipeline()

    print("done pipeline")
    # runner = TaskRunner(pipeline, False)
    #runner.execute()