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
"""Script that generates YAML pipeline for HTTomo using YAML templates from httomo-backends 
(should be installed in environment).

Please run the generator as:
    python -m yaml_pipelines_generator -i /path/to/pipelines.yml -o /path/to/output/
"""
import argparse
import os
from typing import Any, List, Dict
import yaml
import ruamel.yaml

CS = ruamel.yaml.comments.CommentedSeq  # defaults to block style


def FS(x):  # flow style list
    res = CS(x)
    res.fa.set_flow_style()
    return res


import httomo_backends


def yaml_pipelines_generator(
    path_to_pipelines: str, path_to_httomobackends: str, path_to_output_file: str
) -> int:
    """function that builds YAML pipeline using YAML templates from httomo-backends

    Args:
        path_to_pipelines: path to the YAML file which contains a high-level description of the required pipeline to be built.
        path_to_httomobackends: path to httomo-backends on the system, where YAML templates stored.
        path_to_output_file: path to output file with the generated pipeline

    Returns:
        returns zero if the processing is successful
    """

    # open YAML file to inspect
    with open(path_to_pipelines, "r") as stream:
        try:
            pipeline_file_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open(path_to_output_file, "w") as f:
        # a loop over methods in the high-level pipeline file
        methods_no = len(pipeline_file_content)
        pipeline_full = CS()
        for i in range(methods_no):
            method_content = pipeline_file_content[i]
            method_name = method_content["method"]
            module_name = method_content["module_path"]
            # get the corresponding yaml template from httomo-backends
            backend_name = module_name[0 : module_name.find(".")]
            full_path_to_yamls = (
                path_to_httomobackends
                + "/yaml_templates/"
                + backend_name
                + "/"
                + module_name
                + "/"
                + method_name
                + ".yaml"
            )
            with open(full_path_to_yamls, "r") as stream:
                try:
                    yaml_template_method = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            if "loaders" in module_name:
                # should be the first method in the list
                pipeline_full.yaml_set_comment_before_after_key(
                    i,
                    "Standard tomography loader for NeXus files",
                    indent=0,
                )
            elif "rotation" in module_name:
                pipeline_full.yaml_set_comment_before_after_key(
                    i,
                    "Center of Rotation method for automatic center finding. Required for reconstruction",
                    indent=0,
                )
            elif "corr" in module_name and "remove_outlier" in method_name:
                pipeline_full.yaml_set_comment_before_after_key(
                    i,
                    "Removing dead pixels in the data, aka zingers. Only required if there are sharp streaks in the reconstruction.",
                    indent=0,
                )
            elif "normalize" in module_name:
                pipeline_full.yaml_set_comment_before_after_key(
                    i,
                    "Normalisation of projection data using collected flats/darks and taking negative log (not needed with Paganin). ",
                    indent=0,
                )
            elif "stripe" in module_name:
                pipeline_full.yaml_set_comment_before_after_key(
                    i,
                    "Method to remove stripe artefacts in the data that lead to ring artefacts in the reconstruction. ",
                    indent=0,
                )
            elif "algorithm" in module_name:
                pipeline_full.yaml_set_comment_before_after_key(
                    i,
                    "Reconstruction method. Use reference to the center if the method is used above or set to an  ",
                    indent=0,
                )
                # pipeline_full.yaml_add_eol_comment(
                #     "End-of-line comment",
                #     "center",
                #     column=10,
                # )
            elif "calculate_stats" in method_name:
                pipeline_full.yaml_set_comment_before_after_key(
                    i,
                    "Calculate global statistics on the reconstructed volume (min/max needed specifically for data rescaling) ",
                    indent=0,
                )
            else:
                pipeline_full.yaml_set_comment_before_after_key(
                    i,
                    "--------------------------------------------------------#",
                    indent=0,
                )
            pipeline_full += yaml_template_method

        ruamel.yaml.dump(
            pipeline_full,
            f,
            Dumper=ruamel.yaml.RoundTripDumper,
            default_flow_style=False,
            width=50,
            indent=0,
        )

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
        help="Full path to the yaml file with the generated pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    path_to_httomobackends = os.path.dirname(httomo_backends.__file__)
    args = get_args()
    path_to_pipelines = args.input
    path_to_output_file = args.output
    return_val = yaml_pipelines_generator(
        path_to_pipelines, path_to_httomobackends, path_to_output_file
    )
    if return_val == 0:
        print("YAML pipeline has been successfully generated!")
