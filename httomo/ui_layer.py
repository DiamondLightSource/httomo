import yaml
from typing import Any, Dict, List, Optional, Protocol, Tuple
from importlib import import_module, util
from pathlib import Path
import os
import re

from mpi4py import MPI
from mpi4py.MPI import Comm

from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.pipeline import Pipeline

from httomo.method_wrappers import make_method_wrapper
from httomo.runner.loader import LoaderInterface, make_loader
from httomo.runner.output_ref import OutputRef


class UiLayer:
    """A common user interface for different front-ends in httomo.
    We currently support YAML and Python interfaces, but in future
    different UI's can be added based on other data formats, e.g. JSON.
    """

    def __init__(
        self,
        tasks_file_path: Path,
        in_data_file_path: Path,
        comm: Comm,
    ):
        self.repo = MethodDatabaseRepository()
        self.tasks_file_path = tasks_file_path
        self.in_data_file = in_data_file_path
        self.comm = comm

        root, ext = os.path.splitext(self.tasks_file_path)
        if ext.upper() in [".YAML", ".YML"]:
            # loading yaml file with tasks provided
            self.PipelineStageConfig = _yaml_loader(self.tasks_file_path)
        elif ext.upper() == ".PY":
            # loading python file with tasks provided
            self.PipelineStageConfig = _python_tasks_loader(self.tasks_file_path)
        else:
            raise ValueError(
                f"The extension {ext} of the file {root} with tasks is unknown."
            )

    def build_pipeline(self) -> Pipeline:
        side_outputs_collect: List[
            Tuple[int, str, dict]
        ] = []  # saves [task_no, id, side_outputs] for tasks with side_outputs
        methods_list: List[MethodWrapper] = []
        loader: Optional[LoaderInterface] = None
        for task_no, task_conf in enumerate(self.PipelineStageConfig):
            parameters = task_conf.get("parameters", dict())
            if "loaders" in task_conf["module_path"]:
                parameters["in_file"] = self.in_data_file
                # unpack params and initiate a loader
                loader = make_loader(
                    self.repo,
                    task_conf["module_path"],
                    task_conf["method"],
                    self.comm,
                    **parameters,
                )
            else:
                if "side_outputs" not in task_conf:
                    task_conf["side_outputs"] = {}
                else:
                    side_outputs_collect.append(
                        (task_no, task_conf["id"], task_conf["side_outputs"])
                    )
                _update_side_output_references(
                    parameters, side_outputs_collect, methods_list
                )
                # unpack params of a method and append to a list of methods
                method = make_method_wrapper(
                    method_repository=self.repo,
                    module_path=task_conf["module_path"],
                    method_name=task_conf["method"],
                    comm=self.comm,
                    save_result=task_conf.get("save_result", None),
                    output_mapping=task_conf["side_outputs"],
                    **parameters,
                )
                methods_list.append(method)
        if loader is None:
            raise ValueError("Got pipeline with no loader")
        return Pipeline(loader=loader, methods=methods_list)


def _update_side_output_references(
    parameters: dict,
    side_outputs_collect: List[Tuple[int, str, dict]],
    methods_list: List[MethodWrapper],
):
    pattern = re.compile(r"^\$\{\{([A-Za-z0-9_.]+)\}\}$")
    # check if there is a reference to side_outputs to cross-link
    for key, value in parameters.items():
        if not isinstance(value, str) or value is None:
            continue
        result_extr = pattern.search(value)
        if result_extr is None:
            continue
        internal_expression = result_extr.group(1)
        (ref_id, side_str, ref_arg) = internal_expression.split(".")
        if side_str != "side_outputs":
            raise ValueError(
                "Config error: output references must be of the form <id>.side_outputs.<name>"
            )

        # lets find the referred id in "side_outputs_collect"
        try:
            item = next(filter(lambda x: x[1] == ref_id, side_outputs_collect))
        except StopIteration:
            raise ValueError(
                f"could not find side output referenced by {internal_expression}"
            )

        # refer to methods_list[items[0]-1]
        parameters[key] = OutputRef(methods_list[item[0] - 1], ref_arg)


def _yaml_loader(file_path: Path) -> list:
    with open(file_path, "r") as f:
        tasks_list = list(yaml.load_all(f, Loader=yaml.FullLoader))
    return tasks_list[0]


def _python_tasks_loader(file_path: Path) -> list:
    module_spec = util.spec_from_file_location("methods_to_list", file_path)
    if module_spec is None:
        raise ValueError("Could not read module spec")
    foo = util.module_from_spec(module_spec)
    if module_spec.loader is None:
        raise ValueError("Module spec has no loader")
    module_spec.loader.exec_module(foo)
    tasks_list = list(foo.methods_to_list())
    return tasks_list
