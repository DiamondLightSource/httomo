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
# Created Date: 13/April/2023
# version ='0.1'
# ---------------------------------------------------------------------------
"""After _all_ templates have been generated for TomoPy we need to remove the ones
that are not currently supported by httomo. We do that by looking into
the library file for TomoPy."""

import argparse
import importlib
import inspect
import os
import re
import shutil
from pathlib import Path

import yaml


def templates_filter(path_to_modules: str, library_file: str) -> int:
    """function that removes unsupported by httomo YAML templates in TomoPy

    Args:
        path_to_modules (str): path to the list of modules yaml file
        library_file (str): path to the library with the supported functions of TomoPy

    Returns:
        int: returns zero if the processing is succesfull
    """
    software_name = "tomopy"
    yaml_info_path = Path(library_file)
    if not yaml_info_path.exists():
        err_str = f"The YAML file {yaml_info_path} doesn't exist."
        raise ValueError(err_str)

    with open(yaml_info_path, "r") as f:
        yaml_library = yaml.safe_load(f)

    methods_list: list = []
    for module, module_dict in yaml_library.items():
        for module2, module_dict2 in module_dict.items():
            for method_name in module_dict2:
                methods_list.append(method_name)

    subfolders = [f.path for f in os.scandir(path_to_modules) if f.is_dir()]
    for folder in subfolders:
        for filename in os.listdir(folder):
            filename_short = filename.split(".")
            if filename_short[0] not in methods_list:
                print(f"Removed template: {filename_short[0]}")
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))
    return 0


def get_args():
    parser = argparse.ArgumentParser(
        description="Removes unsupported by httomo templates in TomoPy."
    )
    parser.add_argument(
        "-t",
        "--templates",
        type=str,
        default=None,
        help="A path to the folder where generated templates stored.",
    )
    parser.add_argument(
        "-l",
        "--library",
        type=str,
        default=None,
        help="A path to the library YAML file with the supported functions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    current_dir = os.path.basename(os.path.abspath(os.curdir))
    args = get_args()
    path_to_modules = args.templates
    library_file = args.library
    return_val = templates_filter(path_to_modules, library_file)
    if return_val == 0:
        print("The templates have been filtered!")
