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
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>
# Created Date: 21 September 2023
# ---------------------------------------------------------------------------
"""Modules for memory estimation for morph functions"""

import math
from typing import Tuple
import numpy as np

from httomo.runner.output_ref import OutputRef

__all__ = [
    "_calc_memory_bytes_data_resampler",
    "_calc_output_dim_data_resampler",
    "_calc_memory_bytes_sino_360_to_180",
    "_calc_output_dim_sino_360_to_180",
]


def _calc_output_dim_data_resampler(non_slice_dims_shape, **kwargs):
    return kwargs["newshape"]


def _calc_memory_bytes_data_resampler(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    newshape = kwargs["newshape"]
    interpolation = kwargs["interpolation"]

    input_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    xi = 2 * np.prod(newshape) * dtype.itemsize
    output_size = np.prod(newshape) * dtype.itemsize

    # interpolation happens in 2d so we should allocate for it, the exact value is unknown
    if interpolation == "nearest":
        interpolator = 3 * (input_size + output_size)
    if interpolation == "linear":
        interpolator = 4 * (input_size + output_size)

    tot_memory_bytes = input_size + output_size + interpolator
    return (tot_memory_bytes, xi)


def _calc_output_dim_sino_360_to_180(
    non_slice_dims_shape: Tuple[int, int],
    **kwargs,
) -> Tuple[int, int]:
    assert "overlap" in kwargs, "Expected overlap in method parameters"
    overlap_side_output = kwargs["overlap"]
    assert isinstance(
        overlap_side_output, OutputRef
    ), "Expected overlap to be in an OutputRef"
    overlap: float = overlap_side_output.value

    original_sino_width = non_slice_dims_shape[1]
    stitched_sino_width = original_sino_width * 2 - math.ceil(overlap)
    return non_slice_dims_shape[0] // 2, stitched_sino_width


def _calc_memory_bytes_sino_360_to_180(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    assert "overlap" in kwargs, "Expected overlap in method parameters"
    overlap_side_output = kwargs["overlap"]
    assert isinstance(
        overlap_side_output, OutputRef
    ), "Expected overlap to be in an OutputRef"
    overlap: float = overlap_side_output.value

    original_sino_width = non_slice_dims_shape[1]
    stitched_sino_width = original_sino_width * 2 - math.ceil(overlap)
    n = non_slice_dims_shape[0] // 2
    stitched_non_slice_dims = (n, stitched_sino_width)

    input_slice_size = int(np.prod(non_slice_dims_shape)) * dtype.itemsize
    output_slice_size = int(np.prod(stitched_non_slice_dims)) * dtype.itemsize

    summand_shape: Tuple[int, int] = (n, int(overlap))
    # Multiplication between a subset of the original data (`float32`) and the 1D weights array
    # (`float32`) causes a new array to be created that has dtype `float32` (for example,
    # multiplications like `weights * data[:n, :, -overlap]`
    summand_size = int(np.prod(summand_shape)) * np.float32().itemsize

    total_memory_bytes = (
        input_slice_size
        + output_slice_size
        # In both the `if` branch and the `else` branch checking the `rotation` variable value,
        # in total, there are 4 copies of subsets of the `data` array that are made (note that
        # the expressions below are referencing the `if` branch and are slightly different in
        # the `else` branch):
        # 1. fancy indexing: `data[n : 2 * n, :, overlap:][:, :, ::-1]`
        # 2. multiplication: `weights * data[:n, :, :overlap]`
        # 3. multiplication: `weights * data[n : 2 * n, :, :overlap]`
        # 4. fancy indexing (performed on the result of 3):
        # `(weights * data[n : 2 * n, :, :overlap])[:, :, ::-1]`
        + 4 * summand_size
    )

    return total_memory_bytes, 0
