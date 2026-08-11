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
# Created Date: 11/August/2026
# version ='0.1'
# ---------------------------------------------------------------------------
"""This script modifies a given pipeline by changing parameters in it."""

import argparse
import yaml
from typing import Union


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that modifies parameters in a YAML pipeline for HTTomo "
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./",
        help="Full path to a specific pipeline that needs modification.",
    )
    return parser.parse_args()


def change_value_parameters_method_pipeline(
    yaml_path: str,
    method: list,
    key: list,
    value: list,
    save_result: Union[None, bool] = None,
):
    # changes methods parameters in the given pipeline and re-save the pipeline
    with open(yaml_path, "r") as f:
        conf = list(yaml.load_all(f, Loader=yaml.FullLoader))
    opened_yaml = conf[0]
    methods_no = len(opened_yaml)
    methods_no_correct = len(method)
    for i in range(methods_no):
        method_content = opened_yaml[i]
        method_name = method_content["method"]
        for j in range(methods_no_correct):
            if method[j] == method_name:
                # change something in parameters here
                opened_yaml[i]["parameters"][key[j]] = value[j]
                if save_result is not None:
                    # add save_result to the list of keys
                    opened_yaml[i]["save_result"] = save_result

    with open(yaml_path, "w") as file_descriptor:
        yaml.dump(
            opened_yaml, file_descriptor, default_flow_style=False, sort_keys=False
        )


if __name__ == "__main__":
    args = get_args()
    path_to_pipeline = args.input

    change_value_parameters_method_pipeline(
        path_to_pipeline,
        method=[
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "standard_tomo",
            "find_center_vo",
        ],
        key=[
            "preview",
            "data_path",
            "image_key_path",
            "rotation_angles",
            "darks",
            "flats",
            "ind",
        ],
        value=[
            {"detector_y": {"start": 500, "stop": 510}},
            "/exchange/data",
            None,
            {
                "user_defined": {
                    "start_angle": 0,
                    "stop_angle": 179.876,
                    "angles_total": 1500,
                }
            },
            {
                "file": "input_data",
                "image_key_path": None,
                "data_path": "/exchange/data_dark",
            },
            {
                "file": "input_data",
                "image_key_path": None,
                "data_path": "/exchange/data_white",
            },
            "mid",
        ],
    )

    message_str = f"Pipeline {path_to_pipeline} has been modified."
    print(message_str)
