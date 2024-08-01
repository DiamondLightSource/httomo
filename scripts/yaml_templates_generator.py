#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : Tomography Team <scientificsoftware@diamond.ac.uk>
# Created Date: 14/October/2022
# version ='0.3'
# ---------------------------------------------------------------------------
"""Script that exposes all functions of a given software package as YAML templates.

Please run the generator as:
    python -m yaml_templates_generator -i /path/to/modules.yml -o /path/to/output/
"""
import argparse
import importlib
import inspect
import os
import re
from typing import Any, List, Dict

import yaml


def yaml_generator(path_to_modules: str, output_folder: str) -> int:
    """function that exposes all method of a given software package as YAML templates

    Args:
        path_to_modules: path to the list of modules yaml file
        output_folder: path to output folder with saved templates

    Returns:
        returns zero if the processing is succesfull
    """
    discard_keys = _get_discard_keys()
    no_data_out_modules = _get_discard_data_out()

    # open YAML file with modules to inspect
    with open(path_to_modules, "r") as stream:
        try:
            modules_list = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # a loop over modules in the file
    modules_no = len(modules_list)
    for i in range(modules_no):
        module_name = modules_list[i]
        try:
            imported_module = importlib.import_module(str(module_name))
        except NameError:
            print(
                "Import of the module {} has failed, check if software installed".format(
                    module_name
                )
            )
        methods_list = imported_module.__all__  # get all the methods in the module
        methods_no = len(methods_list)

        # a loop over all methods in the module
        for m in range(methods_no):
            method_name = methods_list[m]
            print("Inspecting the signature of the {} method".format(method_name))
            get_method_params = inspect.signature(
                getattr(imported_module, methods_list[m])
            )
            # get method docstrings
            get_method_docs = inspect.getdoc(getattr(imported_module, methods_list[m]))

            # put the parameters in the dictionary
            params_list: List = []
            params_dict: Dict = {}
            for name, value in get_method_params.parameters.items():
                if value is not None:
                    append = True
                    for x in discard_keys:
                        if name == x:
                            append = False
                            break
                    if append:
                        _set_param_value(name, value, params_dict)
            method_dict = {
                "method": method_name,
                "module_path": module_name,
                "parameters": params_dict,
            }
            _set_dict_special_cases(method_dict, method_name)
            params_list = [method_dict]
            _save_yaml(module_name, method_name, params_list)
    return 0


def _set_param_value(name: str, value: inspect.Parameter, params_dict: Dict[str, Any]):
    """Set param value for method inside dictionary
    Args:
        name: Parameter name
        value: Parameter value
        params_dict: Dict containing method's parameter names and values
    """
    if value.default is inspect.Parameter.empty and name != "kwargs":
        if name in ["proj1", "proj2"]:
            params_dict[name] = "auto"
        else:
            params_dict[name] = "REQUIRED"
    elif name == "kwargs":
        # params_dict["#additional parameters"] = "AVAILABLE"
        # parsing hashtag to yaml comes with quotes, for now we simply ignore the field
        pass
    elif name == "axis":
        params_dict[name] = "auto"
    elif name == "asynchronous":
        params_dict[name] = True
    elif name == "center":
        # Temporary value
        params_dict[name] = "${{centering.side_outputs.centre_of_rotation}}"
    elif name == "glob_stats":
        params_dict[name] = "${{statistics.side_outputs.glob_stats}}"
    elif name == "overlap":
        params_dict[name] = "${{centering.side_outputs.overlap}}"
    else:
        params_dict[name] = value.default


def _save_yaml(module_name: str, method_name: str, params_list: List[str]):
    """Save the list as a YAML file
    Args:
        module_name: Name of module
        method_name: Name of method
        params_list: List of parameters
    """
    path_dir = output_folder + "/" + module_name
    path_file = path_dir + "/" + str(method_name) + ".yaml"

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    with open(path_file, "w") as file:
        outputs = yaml.dump(params_list, file, sort_keys=False)


def _set_dict_special_cases(method_dict: Dict, method_name: str):
    """Dealing with special cases for "data_out"

    Args:
        method_dict: Dictionary of modules and parameters
        method_name: Name of method
    """
    if method_name in ["find_center_vo", "find_center_pc"]:
        method_dict["id"] = "centering"
        method_dict["side_outputs"] = {"cor": "centre_of_rotation"}
    if method_name in "find_center_360":
        method_dict["id"] = "centering"
        method_dict["side_outputs"] = {
            "cor": "centre_of_rotation",
            "overlap": "overlap",
            "side": "side",
            "overlap_position": "overlap_position",
        }
    if method_name in "calculate_stats":
        method_dict["id"] = "statistics"
        method_dict["side_outputs"] = {"glob_stats": "glob_stats"}


def _get_discard_data_out() -> List[str]:
    """Discard data_out from certain modules

    Returns: list of data_out to discard
    """
    discard_data_out = ["save_to_images"]
    return discard_data_out


def _get_discard_keys() -> List[str]:
    """Can work with any software in principle,
    but for TomoPy and httomolib there are additional keys
    that needed to be discarded in templates in order to let
    httomo work smoothly.

    Returns: List of keys to discard
    """
    discard_keys = [
        "in_file",
        "data_in",
        "tomo",
        "arr",
        "prj",
        "data",
        "ncore",
        "nchunk",
        "flats",
        "flat",
        "dark",
        "darks",
        "theta",
        "out",
        "ang",
        "comm_rank",
        "out_dir",
        "angles",
        "gpu_id",
        "comm",
        "offset",
    ]
    return discard_keys


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that exposes all functions "
        "of a given software package as YAML templates."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="A path to the list of modules yaml file"
        "which is needed to be inspected and functions extracted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="Directory to save the yaml templates in.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    current_dir = os.path.basename(os.path.abspath(os.curdir))
    args = get_args()
    path_to_modules = args.input
    output_folder = args.output
    return_val = yaml_generator(path_to_modules, output_folder)
    if return_val == 0:
        print("The methods as YAML templates have been successfully generated!")
