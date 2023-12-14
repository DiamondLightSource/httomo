import yaml
from typing import Any, Dict, List, Protocol, TypeAlias
from importlib import import_module, util
import os
import re

from mpi4py import MPI
from mpi4py.MPI import Comm

from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.pipeline import Pipeline

from httomo.runner.backend_wrapper import make_backend_wrapper
from httomo.runner.loader import make_loader
from httomo.runner.output_ref import OutputRef

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
        if ext in ['.yaml', '.yaml'.upper()]:
            # loading yaml file with tasks provided
            self.PipelineStageConfig = _yaml_loader(self.tasks_file_path)[0]
        elif ext in ['.py', '.py'.upper()]:
            # loading python file with tasks provided
            self.PipelineStageConfig = _python_tasks_loader(self.tasks_file_path)
        else:
            raise ValueError(
                f"The extension {ext} of the file {root} with tasks is unknown."
            )

    def build_pipeline(self) -> Pipeline:
        side_outputs_collect: list = [] # saves [task_no, id, side_outputs] for tasks with side_outputs
        methods_list: list = []
        for task_no, task_conf in enumerate(self.PipelineStageConfig):
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
                if "side_outputs" not in task_conf:
                    task_conf['side_outputs'] = {}
                else:
                    side_outputs_collect.append([task_no, task_conf['id'], task_conf['side_outputs']])
                # check if there is a reference to side_outputs to cross-link
                for key, value in task_conf['parameters'].items():
                    if isinstance(value, str) and value is not None:
                        if value.find('${{') != -1:                            
                            result_extr = re.search(r"\{([A-Za-z0-9_.]+)\}", value)
                            internal_expression = result_extr.group(1)
                            (ref_id, side_str, ref_arg) = internal_expression.split(".")
                            # lets find the referred id in "side_outputs_collect"
                            for items in side_outputs_collect:
                                if items[1] == ref_id:
                                    # refer to methods_list[items[0]-1]
                                    task_conf['parameters'][key] = OutputRef(methods_list[items[0]-1], ref_arg)
                # unpack params of a method and append to a list of methods
                method = make_backend_wrapper(
                    self.repo,
                    task_conf['module_path'],
                    task_conf['method'],
                    self.comm,
                    task_conf['side_outputs'],
                    **task_conf['parameters'],
                )
                methods_list.append(method)
        return Pipeline(
            loader=loader,
            methods=methods_list,
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