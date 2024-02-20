from itertools import islice
import yaml
from typing import Any, Dict, List, Optional, Protocol, Tuple
from importlib import import_module, util
from pathlib import Path
import os
import re

import h5py
from mpi4py import MPI
from mpi4py.MPI import Comm
from httomo.darks_flats import DarksFlatsFileConfig

from httomo.methods_database.query import MethodDatabaseRepository
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.pipeline import Pipeline

from httomo.method_wrappers import make_method_wrapper
from httomo.loaders import make_loader
from httomo.runner.loader import LoaderInterface
from httomo.runner.output_ref import OutputRef
from httomo.transform_loader_params import parse_angles, parse_preview


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
        repo=MethodDatabaseRepository(),
    ):
        self.repo = repo
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
        loader = self._setup_loader()

        # saves map {task_id -> method} map
        methods_list: List[MethodWrapper] = []
        method_id_map: Dict[str, MethodWrapper] = dict()
        for i, task_conf in enumerate(self.PipelineStageConfig[1:]):
            parameters = task_conf.get("parameters", dict())
            _update_side_output_references(parameters, method_id_map)
            # unpack params of a method and append to a list of methods
            method = make_method_wrapper(
                method_repository=self.repo,
                module_path=task_conf["module_path"],
                method_name=task_conf["method"],
                comm=self.comm,
                save_result=task_conf.get("save_result", None),
                output_mapping=task_conf.get("side_outputs", dict()),
                task_id=task_conf.get("id", f"task_{i+1}"),
                **parameters,
            )
            if method.task_id in method_id_map:
                raise ValueError(f"duplicate id {method.task_id} in pipeline")
            method_id_map[method.task_id] = method
            methods_list.append(method)

        return Pipeline(loader=loader, methods=methods_list)

    def _setup_loader(self) -> LoaderInterface:
        task_conf = self.PipelineStageConfig[0]
        if "loaders" not in task_conf["module_path"]:
            raise ValueError("Got pipeline with no loader (must be first method)")
        parameters = task_conf.get("parameters", dict())
        parameters["in_file"] = self.in_data_file
        # the following will raise KeyError if not present
        in_file = parameters["in_file"]
        data_path = parameters["data_path"]
        # these will have defaults if not given
        image_key_path = parameters.get("image_key_path", None)
        darks: dict = parameters.get("darks", dict())
        darks_file = darks.get("file", in_file)
        darks_path = darks.get("data_path", data_path)
        darks_image_key = darks.get("image_key_path", image_key_path)
        flats: dict = parameters.get("flats", dict())
        flats_file = flats.get("file", in_file)
        flats_path = flats.get("data_path", data_path)
        flats_image_key = flats.get("image_key_path", image_key_path)
        angles = parse_angles(parameters["rotation_angles"])

        with h5py.File(in_file, "r") as f:
            data_shape = f[data_path].shape
        preview = parse_preview(parameters["preview"], data_shape)

        loader = make_loader(
            repo=self.repo,
            module_path=task_conf["module_path"],
            method_name=task_conf["method"],
            in_file=Path(in_file),
            data_path=data_path,
            image_key_path=image_key_path,
            angles=angles,
            darks=DarksFlatsFileConfig(
                file=darks_file, data_path=darks_path, image_key_path=darks_image_key
            ),
            flats=DarksFlatsFileConfig(
                file=flats_file, data_path=flats_path, image_key_path=flats_image_key
            ),
            preview=preview,
            comm=self.comm,
        )

        return loader


def _update_side_output_references(
    parameters: dict, method_id_map: Dict[str, MethodWrapper]
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

        method = method_id_map.get(ref_id, None)
        if method is None:
            raise ValueError(
                f"could not find method referenced by {internal_expression}"
            )

        parameters[key] = OutputRef(method, ref_arg)


def _yaml_loader(file_path: Path) -> list:
    with open(file_path, "r") as f:
        tasks_list = list(yaml.load_all(f, Loader=yaml.FullLoader))
    return tasks_list[0]


def _python_tasks_loader(file_path: Path) -> list:
    module_spec = util.spec_from_file_location("methods_to_list", file_path)
    assert module_spec is not None, "error reading module spec"
    foo = util.module_from_spec(module_spec)
    assert module_spec.loader is not None, "module spec has no loader"
    module_spec.loader.exec_module(foo)
    tasks_list = list(foo.methods_to_list())
    return tasks_list
