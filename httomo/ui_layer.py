import yaml
from typing import Any, Dict, List, Protocol, TypeAlias
from importlib import import_module, util
from pathlib import Path
import os
import re

from mpi4py import MPI
from mpi4py.MPI import Comm

from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.pipeline import Pipeline

from httomo.runner.backend_wrapper import make_backend_wrapper
from httomo.runner.loader import Loader, make_loader
from httomo.runner.output_ref import OutputRef

MethodConfig: TypeAlias = Dict[str, Any]
PipelineConfig: TypeAlias = List[MethodConfig]


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
        if ext in [".yaml", ".yaml".upper()]:
            # loading yaml file with tasks provided
            self.PipelineStageConfig = yaml_loader(self.tasks_file_path)
        elif ext in [".py", ".py".upper()]:
            # loading python file with tasks provided
            self.PipelineStageConfig = _python_tasks_loader(self.tasks_file_path)
        else:
            raise ValueError(
                f"The extension {ext} of the file {root} with tasks is unknown."
            )

    def build_pipeline(self) -> Pipeline:
        side_outputs_collect: list = []
        save_result_collect: list = []
        methods_list: list = []
        for task_no, task_conf in enumerate(self.PipelineStageConfig):
            if "loaders" in task_conf["module_path"]:
                loader = self._initiate_loader(task_conf)
            else:
                if "parameters" not in task_conf:
                    task_conf["parameters"] = {}
                _append_save_res(task_conf, save_result_collect)
                _append_side_outputs(task_no, task_conf, side_outputs_collect)
                valid_refs = get_valid_ref_ids(task_conf)
                for k, v in valid_refs.items():
                    (ref_id, side_str, ref_arg) = get_ref_split(v)
                    _save_side_reference(
                        task_conf, side_outputs_collect, methods_list, k, ref_id, ref_arg
                    )
                self._append_methods_list(task_conf, methods_list)
        return Pipeline(
            loader=loader,
            methods=methods_list,
            save_results_set=save_result_collect,
        )

    def _initiate_loader(self, task_conf: MethodConfig) -> Loader:
        """Unpack params and initiate a loader

        Parameters
        ----------
        task_conf
            Dictionary containing method information

        Returns
        -------
        Instance of loader class with method details
        """
        task_conf["parameters"]["in_file"] = self.in_data_file
        loader = make_loader(
            self.repo,
            task_conf["module_path"],
            task_conf["method"],
            self.comm,
            **task_conf["parameters"],
        )
        return loader

    def _append_methods_list(self, task_conf: MethodConfig, methods_list: List) -> None:
        """Unpack params of a method and append to a list of methods

        Parameters
        ----------
        task_conf
            Dictionary containing method parameters
        methods_list
        """
        method = make_backend_wrapper(
            self.repo,
            task_conf["module_path"],
            task_conf["method"],
            self.comm,
            task_conf["side_outputs"],
            **task_conf["parameters"],
        )
        methods_list.append(method)


def _append_save_res(task_conf: MethodConfig, save_result_collect: List) -> None:
    """Appends the save result value inside method dictionary to a list

    Parameters
    ----------
    task_conf
        Dictionary containing method information
    save_result_collect
        List to collect method save result values
    """
    if "save_result" not in task_conf:
        save_result_collect.append(False)
    else:
        save_result_collect.append(task_conf["save_result"])


def _append_side_outputs(
    task_no: int, task_conf: MethodConfig, side_outputs_collect: List
) -> None:
    """Saves [task_no, id, side_outputs] for tasks with side_outputs

    Parameters
    ----------
    task_no
        number of task in pipeline
    task_conf
        Dictionary containing method information
    side_outputs_collect
        side output list
    """
    if "side_outputs" not in task_conf:
        task_conf["side_outputs"] = {}
    else:
        side_outputs_collect.append(
            [task_no, task_conf["id"], task_conf["side_outputs"]]
        )


def get_valid_ref_ids(task_conf: MethodConfig) -> Dict[str, str]:
    """
    Parameters
    ----------
    task_conf
        Dictionary containing method information
    Returns
    -------
    Dictionary of {parameter names: valid reference strings}
    """
    valid_refs = {
        param_name: param_val
        for param_name, param_val in task_conf["parameters"].items()
        if (isinstance(param_val, str)) and (param_val is not None)
        if param_val.find("${{") != -1
    }
    return valid_refs


def get_ref_split(ref_str) -> List:
    """Split the given reference string

    Parameters
    ----------
    ref_str
        reference string

    Returns
    -------
    The internal reference expression split by '.'
    """
    result_extr = re.search(r"\{([A-Za-z0-9_.]+)\}", ref_str)
    internal_expression = result_extr.group(1)
    return internal_expression.split(".")


def _save_side_reference(
    task_conf: MethodConfig,
    side_outputs_collect: List,
    methods_list: List,
    key: str,
    ref_id: str,
    ref_arg: str,
) -> None:
    """Find the reference id in "side_outputs_collect", if it matches, then save

    Parameters
    ----------
    task_conf
        Dictionary containing method parameters
    side_outputs_collect
        List of side (additional) outputs
    methods_list
    key:
        Parameter name
    ref_id:
        Side output reference id
    ref_arg:
        Side output reference str
    """
    for items in side_outputs_collect:
        if items[1] == ref_id:
            # If the side output reference is a match
            task_conf["parameters"][key] = OutputRef(
                methods_list[items[0] - 1], ref_arg
            )


def yaml_loader(
    file_path: Path, loader: yaml.Loader = yaml.FullLoader
) -> PipelineConfig:
    """Loads provided yaml file and returns dict

    Parameters
    ----------
    file_path
        yaml file to load
    loader
        yaml loader to use

    Returns
    -------
    PipelineConfig
    """
    with open(file_path, "r") as f:
        tasks_list = list(yaml.load_all(f, Loader=loader))
    return tasks_list[0]


def _python_tasks_loader(file_path: Path) -> list:
    module_spec = util.spec_from_file_location("methods_to_list", file_path)
    foo = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(foo)
    tasks_list = list(foo.methods_to_list())
    return tasks_list
