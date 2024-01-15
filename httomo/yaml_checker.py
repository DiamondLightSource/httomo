"""
Module for checking the validity of yaml files.
"""
import os
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, TypeAlias

import h5py
import yaml

from httomo.utils import Colour
from httomo.yaml_utils import get_external_package_current_version

from . import __version__

__all__ = [
    "sanity_check",
    "validate_yaml_config",
]

MethodConfig: TypeAlias = Dict[str, Dict[str, Any]]
PipelineStageConfig: TypeAlias = List[MethodConfig]
PipelineConfig: TypeAlias = List[PipelineStageConfig]

def sanity_check(conf_generator: Iterator[Any]) -> bool:
    """
    Check if the yaml file is properly indented, has valid mapping and tags.
    """
    _print_with_colour(
        "Checking that the YAML_CONFIG is properly indented and has valid mappings and tags...",
        colour=Colour.GREEN,
    )
    try:
        # Convert generator into a list, to force all elements to be attempted
        # to be interpreted as python objects, and thus initiating all the YAML
        # parsing checks performed by `yaml`
        list(conf_generator)
        _print_with_colour(
            "Sanity check of the YAML_CONFIG was successfully done...\n",
            colour=Colour.GREEN,
        )
        return True
    except yaml.parser.ParserError as e:
        line = e.problem_mark.line
        _print_with_colour(
            f"Incorrect indentation in the YAML_CONFIG file at line {line}. "
            "Please recheck the indentation of the file."
        )
        return False
    except yaml.scanner.ScannerError as e:
        _print_with_colour(
            f"Incorrect mapping in the YAML_CONFIG file at line {e.problem_mark.line + 1}."
        )
        return False
    except yaml.constructor.ConstructorError as e:
        _print_with_colour(
            f"Invalid tag in the YAML_CONFIG file at line {e.problem_mark.line + 1}."
        )
        return False
    except yaml.reader.ReaderError as e:
        _print_with_colour(
            f"Failed to parse YAML file at line {e.problem_mark.line + 1}: {e}"
        )
        return False
    except yaml.YAMLError as e:
        if hasattr(e, "problem_mark"):
            _print_with_colour(
                f"Error in the YAML_CONFIG file at line {e.problem_mark.line}. "
                "Please recheck the file."
            )
        return False

def check_first_method_is_loader(conf: PipelineConfig) -> bool:
    """
    Check that the first method in pipeline is a
    loader.
    """
    first_stage_method = conf[0][0]
    module_name = list(first_stage_method.keys())[0]

    _print_with_colour(
        "Checking that the first method in the pipeline is a loader...",
        colour=Colour.GREEN,
    )
    if module_name != "httomo.data.hdf.loaders":
        _print_with_colour(
            "The first method in the YAML_CONFIG file is not a loader from "
            "'httomo.data.hdf.loaders'. Please recheck the yaml file."
        )
        return False
    _print_with_colour("Loader check successful!!\n", colour=Colour.GREEN)

    return True


def check_hdf5_paths_against_loader(
        conf: MethodConfig,
        in_file_path: str
) -> bool:
    """
    Check that the hdf5 paths given as parameters to the loader indeed exist in
    the given data file.
    """
    with h5py.File(in_file_path, "r") as f:
        hdf5_members = []
        _store_hdf5_members(f, hdf5_members)
        hdf5_members = [m[0] for m in hdf5_members]

    _print_with_colour(
        "Checking that the paths to the data and keys in the YAML_CONFIG file "
        "match the paths and keys in the input file (IN_DATA)...",
        colour=Colour.GREEN,
    )
    params = conf['parameters']
    _path_keys = [key for key in params if "_path" in key]
    for key in _path_keys:
        if params[key].strip("/") not in hdf5_members:
            _print_with_colour(
                f"'{params[key]}' is not a valid path to a dataset in YAML_CONFIG. "
                "Please recheck the yaml file."
            )
            return False
    _print_with_colour("Loader paths check successful!!\n", colour=Colour.GREEN)
    return True


def check_methods_exist_in_templates(conf: PipelineConfig) -> bool:
    """
    Check if the methods in the pipeline YAML file are valid methods, by
    checking if they exist in the template YAML files.
    """
    modules, methods, packages = _get_pipeline_info(conf)
    template_yaml_files = _get_yaml_templates(modules, methods, packages)

    for i, f in enumerate(template_yaml_files):
        if not os.path.exists(f):
            _print_with_colour(
                f"'{modules[i] + '/' + methods[i]}' is not a valid"
                " path to a method. Please recheck the yaml file."
            )
            return False

    return True


def check_valid_method_parameters(
        conf: PipelineConfig,
) -> bool:
    """
    Check each method config in the pipeline against the templates to see if
    the given parameter names are valid.
    """
    modules, methods, packages = _get_pipeline_info(conf)
    template_yaml_files = _get_yaml_templates(modules, methods, packages)
    template_yaml_data_list: List[Generator] = []

    for f in template_yaml_files:
        with open(f, "r") as template:
            method_dict = yaml.load(template, Loader=yaml.FullLoader)[0]
            template_yaml_data_list.append(
                next(iter(method_dict['parameters']))
            )

    for i, _ in enumerate(modules):
        end_str_list = ["Checking '", methods[i], "' and its parameters..."]
        colours = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
        _print_with_colour(end_str_list, colours)
        d1 = methods[i]
        d2 = template_yaml_data_list[i]

        for key in d1.keys():
            for parameter in d1[key].keys():
                assert isinstance(parameter, str)

                if parameter not in d2[key].keys():
                    _print_with_colour(
                        f"Parameter '{parameter}' in the '{modules[i]}' method is not valid."
                    )
                    return False

                # there should be no REQUIRED parameters in the YAML_CONFIG file
                if d1[key][parameter] == "REQUIRED":
                    _print_with_colour(
                        f"A value is needed for the parameter '{parameter}' in the '{modules[i]}' method."
                        " Please specify a value instead of 'REQUIRED'."
                        " Refer to the method docstring for more information."
                    )
                    return False

                # skip tuples for !Sweep and !SweepRange
                if isinstance(d1[key][parameter], tuple) or None in (
                    d1[key][parameter],
                    d2[key][parameter],
                ):
                    continue

                if not isinstance(d1[key][parameter], type(d2[key][parameter])):
                    _print_with_colour(
                        f"Value assigned to parameter '{parameter}' in the '{next(iter(methods[i]))}' method"
                        f" is not correct. It should be of type {type(d2[key][parameter])}."
                    )
                    return False
    return True


def _get_pipeline_info(conf: PipelineConfig) -> Tuple[List, List, List]:
    """
    Helper function to get modules, methods, and packages in the pipeline YAML.
    """
    modules: List[str] = []
    methods: List[MethodConfig] = []
    for stage in conf:
        for module in stage:
            modules.append(module['module_path'])
            methods.append(module['method'])

    packages = [
        m.split(".")[0] + "/" + get_external_package_current_version(m.split(".")[0])
        if m.split(".")[0] != "httomo"
        else m.split(".")[0] + "/" + __version__
        for m in modules
    ]
    return modules, methods, packages


def _get_yaml_templates(modules: List, methods: List, packages: List) -> List:
    """
    Helper function that fetches YAML template files associated with methods
    passed.
    """
    parent_dir = os.path.dirname(os.path.abspath("__file__"))
    templates_dir = os.path.join(parent_dir, "templates")
    assert os.path.exists(templates_dir)
    return [
        os.path.join(
            templates_dir, packages[i], modules[i], methods[i] + ".yaml"
        )
        for i in range(len(modules))
    ]


def _print_with_colour(end_str: Any, colour: Any = Colour.RED) -> None:
    if isinstance(end_str, list):
        output = "".join(
            [f"{colour}{out}{Colour.END}" for out, colour in zip(end_str, colour)]
        )
        print(output)
    else:
        print(colour + end_str + Colour.END)


def _store_hdf5_members(group, members_list, path=""):
    """store the members of an hdf5 group in a list"""
    for name, value in group.items():
        new_path = f"{path}/{name}" if path else name
        if isinstance(value, h5py.Group):
            _store_hdf5_members(value, members_list, path=new_path)
        elif isinstance(value, h5py.Dataset):
            members_list.append((new_path, value))


def validate_yaml_config(
        yaml_file: os.PathLike,
        in_file: Optional[os.PathLike] = None
) -> bool:
    """
    Check that the modules, methods, and parameters in the `YAML_CONFIG` file
    are valid, and adhere to the same structure as in each corresponding
    module in `httomo.templates`.
    """
    with open(yaml_file, "r") as f:
        conf_generator: Iterator[Any] = yaml.load_all(f, Loader=yaml.FullLoader)
        is_yaml_ok = sanity_check(conf_generator)

    with open(yaml_file, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml.FullLoader))

    # Let all checks run before returning with the result, even if some checks
    # fail, to show all errors present in YAML
    is_first_method_loader = check_first_method_is_loader(conf)
    are_hdf5_paths_correct = True
    if in_file is not None:
        are_hdf5_paths_correct = check_hdf5_paths_against_loader(conf[0][0], str(in_file))
    do_methods_exist = check_methods_exist_in_templates(conf)
    are_method_params_valid = check_valid_method_parameters(conf)

    all_checks_pass = is_yaml_ok and \
        is_first_method_loader and \
        are_hdf5_paths_correct and \
        do_methods_exist and \
        are_method_params_valid

    if not all_checks_pass:
        return False

    end_str = (
        "\nYAML validation successful!! Please feel free to use the `run` "
        "command to run the pipeline."
    )
    _print_with_colour(end_str, colour=Colour.BVIOLET)
    return True
