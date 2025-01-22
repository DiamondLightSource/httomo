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
# Created Date: 22/January/2025
# version ='0.1'
# ---------------------------------------------------------------------------
"""Script that generates YAML pipelines for HTTomo using YAML templates from httomo-backends 
(should be installed in environment).

Please run the generator as:
    python -m yaml_pipelines_generator -i /path/to/pipelines.yml -o /path/to/output/
"""
import argparse
import importlib
import inspect
import os
import re
from typing import Any, List, Dict
import yaml

import httomo_backends


def yaml_pipelines_generator(path_to_pipelines: str, path_to_httomobackends: str, output_folder: str) -> int:
    """function that builds YAML pipelines using YAML templates from httomo-backends

    Args:
        path_to_pipelines: path to the YAML file which contains a high-level description of the required pipeline to be built. 
        path_to_httomobackends: path to httomo-backends on the system, where YAML templates stored. 
        output_folder: path to output folder with saved pipelines

    Returns:
        returns zero if the processing is successful
    """

    return 0


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that generates YAML pipelines for HTTomo "
        "using YAML templates from httomo-backends."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="A path to the list of pipelines needed to be built within a yaml file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="Directory to save the yaml pipelines in.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    path_to_httomobackends = os.path.dirname(httomo_backends.__file__)
    args = get_args()
    path_to_pipelines = args.input
    output_folder = args.output
    return_val = yaml_pipelines_generator(path_to_pipelines, path_to_httomobackends, output_folder)
    if return_val == 0:
        print("YAML pipelines have been successfully generated!")
