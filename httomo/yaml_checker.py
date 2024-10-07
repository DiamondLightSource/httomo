"""
Module for checking the validity of yaml files.
"""

import os
from typing import Any, Dict, Iterator, List, Optional, TypeAlias

import h5py
import httomo_backends as hb
import yaml

from pathlib import Path

from httomo.sweep_runner.param_sweep_yaml_loader import (
    ParamSweepYamlLoader,
    get_param_sweep_yaml_loader,
)
from httomo.ui_layer import (
    get_regex_pattern,
    get_ref_split,
    get_valid_ref_str,
    yaml_loader,
)

__all__ = [
    "sanity_check",
    "validate_yaml_config",
]

MethodConfig: TypeAlias = Dict[str, Any]
PipelineConfig: TypeAlias = List[MethodConfig]


class Colour:
    """
    Class for storing the ANSI escape codes for different colours.
    """

    LIGHT_BLUE = "\033[1;34m"
    LIGHT_BLUE_BCKGR = "\033[1;44m"
    BLUE = "\33[94m"
    CYAN = "\33[96m"
    GREEN = "\33[92m"
    YELLOW = "\33[93m"
    MAGENTA = "\33[95m"
    RED = "\33[91m"
    END = "\033[0m"
    BVIOLET = "\033[1;35m"
    LYELLOW = "\033[33m"
    BACKG_RED = "\x1b[6;37;41m"


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
    first_stage = conf[0]
    module_path = first_stage["module_path"]

    _print_with_colour(
        "Checking that the first method in the pipeline is a loader...",
        colour=Colour.GREEN,
    )
    if module_path != "httomo.data.hdf.loaders":
        _print_with_colour(
            "The first method in the YAML_CONFIG file is not a loader from "
            "'httomo.data.hdf.loaders'. Please recheck the yaml file."
        )
        return False
    _print_with_colour("Loader check successful!!\n", colour=Colour.GREEN)

    return True


def check_hdf5_paths_against_loader(conf: PipelineConfig, in_file_path: str) -> bool:
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
    params = conf[0]["parameters"]
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
    packages = _get_package_info(conf)
    template_yaml_files = _get_template_yaml(conf, packages)

    for i, f in enumerate(template_yaml_files):
        if not os.path.exists(f):
            _print_with_colour(
                f"'{conf[i]['module_path'] + '/' + conf[i]['method']}' is not a valid"
                " method. Please recheck the yaml file."
            )
            return False

    return True


def check_parameter_names_are_known(conf: PipelineConfig) -> bool:
    """
    Check if the parameter name of config methods exists in yaml template method parameters
    """
    template_yaml_conf = _get_template_yaml_conf(conf)
    for method_dict in conf:
        yml_method_list = [
            yml_method_dict
            for yml_method_dict in template_yaml_conf
            if (yml_method_dict["method"] == method_dict["method"])
            and (yml_method_dict["module_path"] == method_dict["module_path"])
        ]
        unknown_param_dict = [
            p
            for yml_method in yml_method_list
            for p, v in method_dict["parameters"].items()
            if p not in yml_method["parameters"].keys()
        ]
        for p in unknown_param_dict:
            _print_with_colour(
                f"Parameter '{p}' in the '{method_dict['method']}' method is not valid."
            )
            return False
    return True


def check_parameter_names_are_str(conf: PipelineConfig) -> bool:
    """Parameter names should be type string"""
    non_str_param_names = {
        method["method"]: param
        for method in conf
        for param, param_value in method["parameters"].items()
        if not isinstance(param, str)
    }
    for method, param in non_str_param_names.items():
        _print_with_colour(
            f"A string is needed for the parameter name '{param}' in the '{method}' method."
            " Refer to the method docstring for more information."
        )
        return False
    return True


def check_no_required_parameter_values(conf: PipelineConfig) -> bool:
    """there should be no REQUIRED parameters in the config pipeline"""
    required_values = {
        method["method"]: param
        for method in conf
        for param, param_value in method["parameters"].items()
        if param_value == "REQUIRED"
    }
    for method, param in required_values.items():
        _print_with_colour(
            f"A value is needed for the parameter '{param}' in the '{method}' method."
            " Please specify a value instead of 'REQUIRED'."
            " Refer to the method docstring for more information."
        )
        return False
    return True


def check_no_imagesaver_after_sweep_method(f: Path) -> bool:
    """check that there shouldn't be image saver present after the sweep method"""
    loader = UniqueKeyLoader
    loader.add_constructor("!Sweep", ParamSweepYamlLoader.sweep_manual)
    loader.add_constructor("!SweepRange", ParamSweepYamlLoader.sweep_range)
    pipeline = yaml_loader(f, loader=loader)

    method_is_sweep = False
    for m in pipeline:
        for value in m["parameters"].values():
            if type(value) is tuple:
                method_is_sweep = True
                sweep_method_name = m["method"]
        if method_is_sweep and m["method"] == "save_to_images":
            _print_with_colour(
                f"This pipeline contains a sweep method ({sweep_method_name}) and also save_to_images method(s)."
                " Please note that the result of the sweep method will be automatically saved as images."
                " Therefore there is no need to add save_to_images after any sweep method, please remove."
            )
            return False
    return True


def check_no_duplicated_keys(f: Path) -> bool:
    """there should be no duplicate keys in yaml file
    Parameters
    ----------
    f
        yaml file to check
    """
    loader = UniqueKeyLoader
    loader.add_constructor("!Sweep", ParamSweepYamlLoader.sweep_manual)
    loader.add_constructor("!SweepRange", ParamSweepYamlLoader.sweep_range)
    try:
        yaml_loader(f, loader=loader)
    except ValueError as e:
        # duplicate key found
        _print_with_colour(str(e), colour=Colour.GREEN)
        return False
    return True


def check_keys(conf: PipelineConfig) -> bool:
    """There should be three main keys in each method"""
    required_keys = ["method", "module_path", "parameters"]
    for method in conf:
        all_keys = method.keys()
        if not all(k in all_keys for k in required_keys):
            missing_keys = set(required_keys) - set(all_keys)
            _print_with_colour(f"Missing keys:")
            print(*missing_keys, sep=", ")
            return False
    return True


def check_id_has_side_out(conf: PipelineConfig) -> bool:
    """Check method with an id has side outputs"""
    method_ids = [m.get("side_outputs") for m in conf if m.get("id")]
    if None in method_ids:
        _print_with_colour(f"A method with an id has no side outputs defined.")
        return False
    return True


def check_ref_id_valid(conf: PipelineConfig) -> bool:
    """Check reference str is matching a valid method id"""
    pattern = get_regex_pattern()
    method_ids = [m.get("id") for m in conf if m.get("id")]
    ref_strs = {
        k: v
        for m in conf
        for k, v in get_valid_ref_str(m.get("parameters", dict())).items()
    }
    for k, v in ref_strs.items():
        (ref_id, side_str, ref_arg) = get_ref_split(v, pattern)
        if ref_id not in method_ids:
            _print_with_colour(
                f"The reference id: {ref_id} was not found to have a matching method id."
            )
            return False
    return True


def check_side_out_matches_ref_arg(conf: PipelineConfig) -> bool:
    """Check reference name exists"""
    pattern = get_regex_pattern()
    ref_strs = {
        k: v
        for m in conf
        for k, v in get_valid_ref_str(m.get("parameters", dict())).items()
    }
    for k, v in ref_strs.items():
        (ref_id, side_str, ref_arg) = get_ref_split(v, pattern)
        side_dicts = [
            m.get(side_str)
            for m in conf
            if m.get("id") == ref_id and m.get(side_str) is not None
        ]
        all_side_out = {k: v for d in side_dicts for k, v in d.items()}
        if ref_arg not in all_side_out.values():
            _print_with_colour(
                f"The reference value: {ref_arg} was not found to have a matching side"
                f"output value."
            )
            return False
    return True


def _get_template_yaml_conf(conf: PipelineConfig) -> PipelineConfig:
    """Get the pipeline config method dictionaries from template yaml files

    Parameters
    ----------
    conf
       The list of method dictionaries in the pipeline
        this specifies which yaml template methods to get

    Returns
    -------
    list of method dictionaries which is loaded from yaml templates
    """
    packages = _get_package_info(conf)
    template_yaml_files = _get_template_yaml(conf, packages)
    template_yaml_conf: PipelineConfig = []
    for f in template_yaml_files:
        tmp_conf = yaml_loader(f)
        # make an assumption there is one method inside each template
        template_yaml_conf.append(tmp_conf[0])
    return template_yaml_conf


def _get_package_info(conf: PipelineConfig) -> List:
    """
    Helper function to get packages from module path.
    """
    modules = [m["module_path"] for m in conf]
    packages = [m.split(".")[0] for m in modules]
    return packages


def _get_template_yaml(conf: PipelineConfig, packages: List) -> List:
    """
    Helper function that fetches template YAML file names associated with methods
    passed.
    """
    httomo_backends_dir = Path(hb.__path__[0])
    templates_dir = httomo_backends_dir / "yaml_templates"
    assert os.path.exists(
        templates_dir
    ), "Dev error: expected YAML templates dir to exist"
    return [
        os.path.join(
            templates_dir,
            packages[i],
            conf[i]["module_path"],
            conf[i]["method"] + ".yaml",
        )
        for i in range(len(conf))
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


def validate_yaml_config(yaml_file: Path, in_file: Optional[Path] = None) -> bool:
    """
    Check that the modules, methods, and parameters in the `YAML_CONFIG` file
    are valid, and adhere to the same structure as in each corresponding
    module in `httomo.yaml_templates`.
    """
    with open(yaml_file, "r") as f:
        conf_generator: Iterator[Any] = yaml.load_all(
            f, Loader=get_param_sweep_yaml_loader()
        )
        is_yaml_ok = sanity_check(conf_generator)

    are_keys_duplicated = check_no_duplicated_keys(yaml_file)
    conf = yaml_loader(yaml_file)

    # Let all checks run before returning with the result, even if some checks
    # fail, to show all errors present in YAML
    is_first_method_loader = check_first_method_is_loader(conf)
    are_hdf5_paths_correct = True
    if in_file is not None:
        are_hdf5_paths_correct = check_hdf5_paths_against_loader(conf, str(in_file))
    do_methods_exist = check_methods_exist_in_templates(conf)
    are_param_names_known = check_parameter_names_are_known(conf)
    are_param_names_type_str = check_parameter_names_are_str(conf)
    id_and_side_out_present = check_id_has_side_out(conf)
    are_ref_ids_valid = check_ref_id_valid(conf)
    side_out_matches_ref_arg = check_side_out_matches_ref_arg(conf)
    required_keys_present = check_keys(conf)
    are_required_parameters_missing = check_no_required_parameter_values(conf)

    all_checks_pass = (
        is_yaml_ok
        and are_keys_duplicated
        and is_first_method_loader
        and are_hdf5_paths_correct
        and do_methods_exist
        and are_param_names_known
        and are_param_names_type_str
        and id_and_side_out_present
        and are_ref_ids_valid
        and side_out_matches_ref_arg
        and required_keys_present
        and are_required_parameters_missing
    )

    if not all_checks_pass:
        return False

    end_str = "\nValidation of pipeline YAML file is successful."
    _print_with_colour(end_str, colour=Colour.BVIOLET)
    return True


class UniqueKeyLoader(yaml.SafeLoader):
    """Check for duplicate keys in yaml"""

    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            each_key = self.construct_object(key_node, deep=deep)
            if each_key in mapping:
                raise ValueError(f"Duplicate Key: {each_key} found{key_node.end_mark}")
            mapping.add(each_key)
        return super().construct_mapping(node, deep)
