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
"""Modules for memory estimation for phase retrieval and phase-contrast enhancement"""

import math
from typing import Tuple
import numpy as np

__all__ = [
    "_calc_memory_bytes_paganin_filter_savu",
    "_calc_memory_bytes_paganin_filter_tomopy",
]


def _calc_memory_bytes_paganin_filter_savu(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    pad_x = kwargs["pad_x"]
    pad_y = kwargs["pad_y"]
    input_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    in_slice_size = (
        (non_slice_dims_shape[0] + 2 * pad_y)
        * (non_slice_dims_shape[1] + 2 * pad_x)
        * dtype.itemsize
    )
    # FFT needs complex inputs, so copy to complex happens first
    complex_slice = in_slice_size / dtype.itemsize * np.complex64().nbytes
    fftplan_slice = complex_slice
    filter_size = complex_slice
    res_slice = np.prod(non_slice_dims_shape) * np.float32().nbytes
    tot_memory_bytes = (
        input_size + in_slice_size + complex_slice + fftplan_slice + res_slice
    )
    return (tot_memory_bytes, filter_size)


def _calc_memory_bytes_paganin_filter_tomopy(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    from httomolibgpu.prep.phase import _shift_bit_length

    # estimate padding size here based on non_slice dimensions
    pad_tup = []
    for index, element in enumerate(non_slice_dims_shape):
        diff = _shift_bit_length(element + 1) - element
        if element % 2 == 0:
            pad_width = diff // 2
            pad_width = (pad_width, pad_width)
        else:
            # need an uneven padding for odd-number lengths
            left_pad = diff // 2
            right_pad = diff - left_pad
            pad_width = (left_pad, right_pad)
        pad_tup.append(pad_width)

    input_size = np.prod(non_slice_dims_shape) * dtype.itemsize

    in_slice_size = (
        (non_slice_dims_shape[0] + pad_tup[0][0] + pad_tup[0][1])
        * (non_slice_dims_shape[1] + pad_tup[1][0] + pad_tup[1][1])
        * dtype.itemsize
    )
    out_slice_size = (
        (non_slice_dims_shape[0] + pad_tup[0][0] + pad_tup[0][1])
        * (non_slice_dims_shape[1] + pad_tup[1][0] + pad_tup[1][1])
        * dtype.itemsize
    )

    # FFT needs complex inputs, so copy to complex happens first
    complex_slice = in_slice_size / dtype.itemsize * np.complex64().nbytes
    fftplan_slice = complex_slice
    grid_size = np.prod(non_slice_dims_shape) * np.float32().nbytes
    filter_size = grid_size
    res_slice = grid_size

    tot_memory_bytes = int(
        input_size
        + in_slice_size
        + out_slice_size
        + 2 * complex_slice
        + 0.5 * fftplan_slice
        + res_slice
    )
    subtract_bytes = int(filter_size + grid_size)

    return (tot_memory_bytes, subtract_bytes)
