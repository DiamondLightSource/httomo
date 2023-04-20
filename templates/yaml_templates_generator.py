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
# Created By  : Daniil Kazantsev <scientificsoftware@diamond.ac.uk>
# Created Date: 14/October/2022
# version ='0.2'
# ---------------------------------------------------------------------------
"""Script that exposes all functions of a given software package as YAML templates.

Please run the generator as:
python -m yaml_templates_generator -m /path/to/modules.yml -o /path/to/output/
"""
import re
import inspect
import yaml
import os
import importlib
import argparse


def yaml_generator(path_to_modules: str, output_folder: str) -> int:
    """function that exposes all method of a given software package as YAML templates

    Args:
        path_to_modules (str): path to the list of modules yaml file
        output_folder (str): path to output folder with saved templates

    Returns:
        int: returns zero if the processing is succesfull
    """
    # Can work with any software in principle,
    # but for TomoPy and httomolib there are additional keys
    # that needed to be discarded in templates in order to let
    # httomo work smoothly.
    discard_keys = [
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
        "glob_stats",
        "comm_rank",
        "out_dir",
        "angles",
        "gpu_id",
    ]  # discard from parameters list

    no_data_out_modules = ["save_to_images"]  # discard data_out from certain modules

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
            print("Import of the module has failed, check if software installed")
        methods_list = imported_module.__all__  # get all the methods in the module
        methods_no = len(methods_list)

        # a loop over all methods in the module
        for m in range(methods_no):
            method_name = methods_list[m]
            get_method_params = inspect.signature(
                getattr(imported_module, methods_list[m])
            )
            # get method docstrings
            get_method_docs = inspect.getdoc(getattr(imported_module, methods_list[m]))

            # put the parameters in the dictionary
            params_list: list = []
            params_dict: dict = {}
            params_dict["data_in"] = "tomo"  # default dataset names
            # dealing with special cases for "data_out"
            if method_name not in no_data_out_modules:
                params_dict["data_out"] = "tomo"
            if method_name in "find_center_vo":
                params_dict["data_out"] = "cor"
            if method_name in "find_center_360":
                params_dict["data_out"] = ["cor", "overlap", "side", "overlap_position"]
            for k, v in get_method_params.parameters.items():
                if v is not None:
                    append = True
                    for x in discard_keys:
                        if str(k) == x:
                            append = False
                            break
                    if append:
                        if str(v).find("=") == -1 and str(k) != "kwargs":
                            params_dict[str(k)] = "REQUIRED"
                        elif str(k) == "kwargs":
                            params_dict["#additional parameters"] = "AVAILABLE"
                        else:
                            params_dict[str(k)] = v.default

            params_list = [{module_name: {method_name: params_dict}}]

            # save the list as a YAML file
            path_dir = output_folder + "/" + module_name
            path_file = path_dir + "/" + str(method_name) + ".yaml"

            if not os.path.exists(path_dir):
                os.makedirs(path_dir)

            with open(path_file, "w") as file:
                outputs = yaml.dump(params_list, file, sort_keys=False)
    return 0


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that exposes all functions "
        "of a given software package as YAML templates."
    )
    parser.add_argument(
        "-m",
        "--modules",
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
    path_to_modules = args.modules
    output_folder = args.output
    return_val = yaml_generator(path_to_modules, output_folder)
    if return_val == 0:
        print("The methods as YAML templates have been succesfully generated!")
