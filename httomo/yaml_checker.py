"""
Module for checking the validity of yaml files.
"""
import os
from typing import Any

import h5py
import yaml

from httomo.utils import Colour
from httomo.yaml_utils import get_external_package_current_version

__all__ = [
    "check_one_method_per_module",
    "sanity_check",
    "validate_yaml_config",
]


def sanity_check(yaml_file):
    """
    Check if the yaml file is properly indented, has valid mapping and tags.
    """
    _print_with_colour(
        "Checking that the YAML_CONFIG is properly indented and has valid mappings and tags...",
        colour=Colour.GREEN,
    )
    with open(yaml_file, "r") as file:
        try:
            yaml_data = open_yaml_config(yaml_file)
            _print_with_colour(
                "Sanity check of the YAML_CONFIG was successfully done...\n",
                colour=Colour.GREEN,
            )
            return yaml_data
        except yaml.parser.ParserError as e:
            line = e.problem_mark.line
            _print_with_colour(
                f"Incorrect indentation in the YAML_CONFIG file at line {line}. "
                "Please recheck the indentation of the file."
            )
        except yaml.scanner.ScannerError as e:
            _print_with_colour(
                f"Incorrect mapping in the YAML_CONFIG file at line {e.problem_mark.line + 1}."
            )
        except yaml.constructor.ConstructorError as e:
            _print_with_colour(
                f"Invalid tag in the YAML_CONFIG file at line {e.problem_mark.line + 1}."
            )
        except yaml.reader.ReaderError as e:
            _print_with_colour(
                f"Failed to parse YAML file at line {e.problem_mark.line + 1}: {e}"
            )
        except yaml.YAMLError as e:
            if hasattr(e, "problem_mark"):
                _print_with_colour(
                    f"Error in the YAML_CONFIG file at line {e.problem_mark.line}. "
                    "Please recheck the file."
                )


def check_one_method_per_module(yaml_file):
    """
    Check that we cannot have a yaml file with more than one method
    being called from one module. For example, we cannot have:

    - tomopy.prep.normalize:
        normalize:
          data_in: tomo
          data_out: tomo
          cutoff: null
        minus_log:
          data_in: tomo
          data_out: tomo
    """
    _print_with_colour(
        "Checking that YAML_CONFIG includes only one method from each module...\n"
        "\nDoing a sanity check first...",
        colour=Colour.GREEN,
    )
    yaml_data = sanity_check(yaml_file)

    lvalues = [value for d in yaml_data for value in d.values()]
    for i, d in enumerate(lvalues):
        assert isinstance(d, dict)
        if len(d) != 1:
            _print_with_colour(
                f"More than one method is being called from the"
                f" module '{next(iter(yaml_data[i]))}'. "
                "Please recheck the yaml file."
            )
            return False

    _print_with_colour(
        "'One method per module' check was also successfully done...\n",
        colour=Colour.GREEN,
    )
    return yaml_data


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


def validate_yaml_config(yaml_file, in_file: str = None) -> bool:
    """
    Check that the modules, methods, and parameters in the `YAML_CONFIG` file
    are valid, and adhere to the same structure as in each corresponding
    module in `httomo.templates`.
    """
    yaml_data = check_one_method_per_module(yaml_file)

    modules = [next(iter(d)) for d in yaml_data]
    methods = [next(iter(d.values())) for d in yaml_data]
    packages = [
        m.split(".")[0] + "/" + get_external_package_current_version(m.split(".")[0])
        if m.split(".")[0] != "httomo"
        else m.split(".")[0]
        for m in modules
    ]

    #: the first method is always a loader
    #: so `testing_pipeline.yaml` should not pass.
    _print_with_colour(
        "Checking that the first method in the pipeline is a loader...",
        colour=Colour.GREEN,
    )
    if modules[0] != "httomo.data.hdf.loaders":
        _print_with_colour(
            "The first method in the YAML_CONFIG file is not a loader from "
            "'httomo.data.hdf.loaders'. Please recheck the yaml file."
        )
        return False
    _print_with_colour("Loader check successful!!\n", colour=Colour.GREEN)

    if in_file is not None:
        with h5py.File(in_file, "r") as f:
            hdf5_members = []
            _store_hdf5_members(f, hdf5_members)
            hdf5_members = [m[0] for m in hdf5_members]

        _print_with_colour(
            "Checking that the paths to the data and keys in the YAML_CONFIG file "
            "match the paths and keys in the input file (IN_DATA)...",
            colour=Colour.GREEN,
        )
        loader_params = next(iter(methods[0].values()))
        _path_keys = [key for key in loader_params if "_path" in key]
        for key in _path_keys:
            if loader_params[key].strip("/") not in hdf5_members:
                _print_with_colour(
                    f"'{loader_params[key]}' is not a valid path to a dataset in YAML_CONFIG. "
                    "Please recheck the yaml file."
                )
                return False
        _print_with_colour("Loader paths check successful!!\n", colour=Colour.GREEN)

    parent_dir = os.path.dirname(os.path.abspath("__file__"))
    templates_dir = os.path.join(parent_dir, "templates")
    assert os.path.exists(templates_dir)

    _template_yaml_files = [
        os.path.join(
            templates_dir, packages[i], modules[i], next(iter(methods[i])) + ".yaml"
        )
        for i in range(len(modules))
    ]

    for i, f in enumerate(_template_yaml_files):
        if not os.path.exists(f):
            _print_with_colour(
                f"'{modules[i] + '/' + next(iter(methods[i]))}' is not a valid"
                " path to a method. Please recheck the yaml file."
            )
            return False

    _template_yaml_data_list = [
        next(iter(d.values()))
        for f in _template_yaml_files
        for d in open_yaml_config(f)
    ]

    for i, _ in enumerate(modules):
        end_str_list = ["Checking '", next(iter(methods[i])), "' and its parameters..."]
        colours = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
        _print_with_colour(end_str_list, colours)
        d1 = methods[i]
        d2 = _template_yaml_data_list[i]

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

    end_str = "\nYAML validation successful!! Please feel free to use the `run` command to run the pipeline."
    _print_with_colour(end_str, colour=Colour.BVIOLET)
    return True
