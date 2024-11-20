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
"""Modules for memory estimation for stripe removal methods"""

import math
from typing import Tuple
import numpy as np

from httomo.cufft import CufftType, cufft_estimate_2d


__all__ = [
    "_calc_memory_bytes_remove_stripe_ti",
    "_calc_memory_bytes_remove_all_stripe",
    "_calc_memory_bytes_raven_filter",
]


def _calc_memory_bytes_remove_stripe_ti(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    # This is admittedly a rough estimation, but it should be about right
    gamma_mem = non_slice_dims_shape[1] * np.float64().itemsize

    in_slice_mem = np.prod(non_slice_dims_shape) * dtype.itemsize
    slice_mean_mem = non_slice_dims_shape[1] * dtype.itemsize * 2
    slice_fft_plan_mem = slice_mean_mem * 3.5
    extra_temp_mem = slice_mean_mem * 8

    tot_memory_bytes = int(
        in_slice_mem + slice_mean_mem + slice_fft_plan_mem + extra_temp_mem
    )
    return (tot_memory_bytes, gamma_mem)


def _calc_memory_bytes_remove_all_stripe(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    # Extremely memory hungry function but it works slice-by-slice so
    # we need to compensate for that.

    input_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    output_size = np.prod(non_slice_dims_shape) * dtype.itemsize

    methods_memory_allocations = int(30 * input_size)

    tot_memory_bytes = int(input_size + output_size)

    return (tot_memory_bytes, methods_memory_allocations)


def _calc_memory_bytes_raven_filter(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:

    pad_x = kwargs["pad_x"]
    pad_y = kwargs["pad_y"]

    input_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    output_size = np.prod(non_slice_dims_shape) * dtype.itemsize

    # Padded input
    padded_non_slice_dims_shape = (
        non_slice_dims_shape[0] + 2 * pad_y,
        non_slice_dims_shape[1] + 2 * pad_x,
    )

    in_slice_size_pad = (
        (padded_non_slice_dims_shape[0])
        * (padded_non_slice_dims_shape[1])
        * dtype.itemsize
    )
    out_slice_size_pad = in_slice_size_pad

    complex_slice_fft_data = in_slice_size_pad / dtype.itemsize * np.complex64().nbytes
    complex_slice_fft_data_shifted = complex_slice_fft_data
    data_out_ifft_complex = complex_slice_fft_data

    # Plan size for 2D FFT
    fftplan_slice = cufft_estimate_2d(
        nx=padded_non_slice_dims_shape[1],
        ny=padded_non_slice_dims_shape[0],
        fft_type=CufftType.CUFFT_C2C,
    )

    tot_memory_bytes = int(
        input_size
        + output_size
        + in_slice_size_pad
        + out_slice_size_pad
        + complex_slice_fft_data
        + complex_slice_fft_data_shifted
        + data_out_ifft_complex
        + 2 * fftplan_slice
    )

    return (tot_memory_bytes, 0)
