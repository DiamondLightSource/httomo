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
# Created Date: 30/January/2025
# version ='0.1'
# ---------------------------------------------------------------------------
"""Executing full-pipeline generation for HTTomo using YAML templates from httomo-backends
and yaml_pipelines_generator script available also in httomo-backends.
"""

import argparse
import os
import glob
import httomo_backends
from httomo_backends.scripts.yaml_pipelines_generator import yaml_pipelines_generator


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that generates YAML pipelines for HTTomo "
        "using YAML templates from httomo-backends."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="Full path to the output pipelines folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    path_to_httomobackends = os.path.dirname(httomo_backends.__file__)
    args = get_args()
    path_to_httomo_pipelines = args.output
    pipelines_folder = path_to_httomobackends + "/pipelines_full/"
    # loop over all pipeline directive files and running the generator
    for filepath in glob.iglob(pipelines_folder + "*.yaml"):
        basename = os.path.basename(filepath)
        outputfile_name = os.path.normpath(basename.replace(r"_directive", r""))
        yaml_pipelines_generator(
            filepath, path_to_httomobackends, path_to_httomo_pipelines + outputfile_name
        )
        message_str = f"{outputfile_name} has been generated."
        print(message_str)
