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
(should be already installed in your environment).

Please run the generator as:
    python -m create_numpy_from_hdf5 -i /path/to/file.hdf5 -o /path/to/output/file.npz
"""
import argparse
import os
import h5py
import numpy as np


def create_numpy_from_hdf5(path_to_hdf5: str, path_to_output_file: str) -> int:
    """
    Args:
        path_to_hdf5: A path to the hdf5 file from which data needs to be extracted.
        path_to_output_file: Output path to the saved dataset as numpy array.

    Returns:
        returns zero if the extraction of data is successful
    """
    h5f = h5py.File(path_to_hdf5, "r")
    path_to_data = "data/"
    proj1 = h5f[path_to_data][0, :, :]  # get the first projection
    dety, detx = np.shape(proj1)

    slices = 10
    projdata_selection = np.empty((slices, dety, detx))
    step = detx // (slices + 2)
    index_prog = step
    for i in range(slices):
        projdata_selection[i, :, :] = h5f[path_to_data][index_prog, :, :]
        index_prog += step
    h5f.close()

    np.savez(path_to_output_file, projdata=projdata_selection)

    return 0


def get_args():
    parser = argparse.ArgumentParser(
        description="Script that creates a numpy array from hdf5"
        "reconstruction file and saves it on disk."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="A path to the hdf5 file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path to the saved dataset as numpy array.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    path_to_hdf5 = args.input
    path_to_output_file = args.output
    return_val = create_numpy_from_hdf5(path_to_hdf5, path_to_output_file)
    if return_val == 0:
        message_str = (
            f"Numpy file {path_to_output_file} has been created from {path_to_hdf5}."
        )
        print(message_str)
