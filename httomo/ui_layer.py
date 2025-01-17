import yaml
from typing import Any, Dict, List, Optional, TypeAlias
from importlib import import_module
from pathlib import Path
import os
import re

import h5py
from mpi4py.MPI import Comm
from httomo.darks_flats import DarksFlatsFileConfig

from httomo.preview import PreviewConfig
from httomo.runner.method_wrapper import MethodWrapper
from httomo.runner.pipeline import Pipeline

from httomo.method_wrappers import make_method_wrapper
from httomo.loaders import make_loader
from httomo.runner.loader import LoaderInterface
from httomo.runner.output_ref import OutputRef
from httomo.sweep_runner.param_sweep_yaml_loader import get_param_sweep_yaml_loader
from httomo.transform_loader_params import parse_angles, parse_preview

from httomo_backends.methods_database.query import MethodDatabaseRepository

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
        repo=MethodDatabaseRepository(),
    ):
        self.repo = repo
        self.tasks_file_path = tasks_file_path
        self.in_data_file = in_data_file_path
        self.comm = comm
        self._preview_config: PreviewConfig | None = None

        # loading yaml file with tasks provided
        self.PipelineStageConfig = yaml_loader(self.tasks_file_path)

    def build_pipeline(self) -> Pipeline:
        loader = self._setup_loader()

        # saves map {task_id -> method} map
        methods_list: List[MethodWrapper] = []
        method_id_map: Dict[str, MethodWrapper] = dict()

        for i, task_conf in enumerate(self.PipelineStageConfig[1:]):
            parameters = task_conf.get("parameters", dict())
            valid_refs = get_valid_ref_str(parameters)
            update_side_output_references(valid_refs, parameters, method_id_map)
            self._append_methods_list(
                i, task_conf, methods_list, parameters, method_id_map
            )
        return Pipeline(loader=loader, methods=methods_list)

    def _append_methods_list(
        self,
        i: int,
        task_conf: MethodConfig,
        methods_list: List,
        parameters: dict,
        method_id_map: Dict[str, MethodWrapper],
    ) -> None:
        """Unpack params of a method and append to a list of methods

        Parameters
        ----------
        i
            item in pipeline config
        task_conf
            dictionary containing method parameters
        methods_list
        parameters
            dictionary of parameters
        method_id_map
            map of methods and ids
        """
        assert (
            self._preview_config is not None
        ), "Preview config should have been stored prior to method wrapper creation"
        method = make_method_wrapper(
            method_repository=self.repo,
            module_path=task_conf["module_path"],
            method_name=task_conf["method"],
            comm=self.comm,
            preview_config=self._preview_config,
            save_result=task_conf.get("save_result", None),
            output_mapping=task_conf.get("side_outputs", dict()),
            task_id=task_conf.get("id", f"task_{i + 1}"),
            **parameters,
        )
        # TODO option to relocate to yaml_checker
        if method.task_id in method_id_map:
            raise ValueError(f"duplicate id {method.task_id} in pipeline")
        method_id_map[method.task_id] = method
        methods_list.append(method)

    def _setup_loader(self) -> LoaderInterface:
        task_conf = self.PipelineStageConfig[0]
        if "loaders" not in task_conf["module_path"]:
            # TODO option to relocate to yaml_checker
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
        preview = parse_preview(parameters.get("preview", None), data_shape)
        self._preview_config = preview

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


def get_valid_ref_str(parameters: Dict[str, Any]) -> Dict[str, str]:
    """Find valid reference strings inside dictionary

    Parameters
    ----------
    parameters
        Dictionary containing parameter information
    Returns
    -------
    Dictionary of {parameter names: valid reference strings}
    """
    valid_refs = {
        param_name: param_val
        for param_name, param_val in parameters.items()
        if (isinstance(param_val, str)) and (param_val is not None)
        if param_val.find("${{") != -1
    }
    return valid_refs


def update_side_output_references(
    valid_refs: Dict[str, Any],
    parameters: Dict[str, Any],
    method_id_map: Dict[str, MethodWrapper],
) -> None:
    """Iterate over valid reference strings, split, check, set

    Parameters
    ----------
    valid_refs
        dict of valid reference id strings {param_name: ref id str}
    parameters
        dict of all parameters
    method_id_map
        map of methods and ids
    """
    pattern = get_regex_pattern()
    # check if there is a reference to side_outputs to cross-link
    for param_name, param_value in valid_refs.items():
        (ref_id, side_str, ref_arg) = get_ref_split(param_value, pattern)
        if ref_id is None:
            continue
        method = method_id_map.get(ref_id, None)
        check_valid_ref_id(side_str, ref_id, param_value, method)
        parameters[param_name] = OutputRef(method, ref_arg)


def get_regex_pattern() -> re.Pattern:
    """Return the reference string regex pattern to search for
    Returns
    -------
    Regex pattern specification
    """
    return re.compile(r"^\$\{\{([A-Za-z0-9_.]+)\}\}$")


def get_ref_split(ref_str: str, pattern: re.Pattern) -> List:
    """Split the given reference string

    Parameters
    ----------
    ref_str
        reference string
    pattern
        regex pattern specification

    Returns
    -------
    The internal reference expression split by '.'
    """
    result_extr = pattern.search(ref_str)
    if result_extr is None:
        return [None] * 3
    internal_expression = result_extr.group(1)
    return internal_expression.split(".", 3)


def check_valid_ref_id(
    side_str: str, ref_id: str, v: str, method: MethodWrapper
) -> None:
    """Check the reference values are valid
    TODO option to relocate to yaml_checker

    Parameters
    ----------
    side_str
        side output string
    ref_id
        reference id
    v
        ref value
    method
        method found from ref
    """
    if side_str != "side_outputs":
        raise ValueError(
            "Config error: output references must be of the form <id>.side_outputs.<name>"
        )
    if method is None:
        raise ValueError(f"could not find method referenced by {ref_id} in {v}")


def yaml_loader(
    file_path: Path, loader: Optional[type[yaml.SafeLoader]] = None
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
    if loader is None:
        loader = get_param_sweep_yaml_loader()
    with open(file_path, "r") as f:
        tasks_list = list(yaml.load_all(f, Loader=loader))
    return tasks_list[0]
