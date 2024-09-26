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

from httomo.cufft import CufftType, cufft_estimate_2d

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

    # Input (unpadded)
    unpadded_in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize

    # Padded input
    padded_non_slice_dims_shape = (
        non_slice_dims_shape[0] + 2 * pad_y,
        non_slice_dims_shape[1] + 2 * pad_x,
    )
    padded_in_slice_size = (
        padded_non_slice_dims_shape[0] * padded_non_slice_dims_shape[1] * dtype.itemsize
    )

    # Padded input cast to `complex64`
    complex_slice = padded_in_slice_size / dtype.itemsize * np.complex64().nbytes

    # Plan size for 2D FFT
    fftplan_slice_size = cufft_estimate_2d(
        nx=padded_non_slice_dims_shape[1],
        ny=padded_non_slice_dims_shape[0],
        fft_type=CufftType.CUFFT_C2C,
    )

    # Shape of 2D filter is the same as the padded `complex64` slice shape, so the size will be
    # the same
    filter_size = complex_slice

    # Size of cropped/unpadded + cast to float32 result of 2D IFFT
    cropped_float32_res_slice = np.prod(non_slice_dims_shape) * np.float32().nbytes

    # If the FFT plan size is negligible for some reason, this changes where the peak GPU
    # memory usage occurs. Hence, the if/else branching below for calculating the total bytes.
    NEGLIGIBLE_FFT_PLAN_SIZE = 16
    if fftplan_slice_size < NEGLIGIBLE_FFT_PLAN_SIZE:
        tot_memory_bytes = int(
            unpadded_in_slice_size + padded_in_slice_size + complex_slice
        )
    else:
        tot_memory_bytes = int(
            unpadded_in_slice_size
            + padded_in_slice_size
            + complex_slice
            # The padded float32 array is deallocated when a copy is made when casting to complex64
            # and the variable `padded_tomo` is reassigned to the complex64 version
            - padded_in_slice_size
            + fftplan_slice_size
            + cropped_float32_res_slice
        )

    return (tot_memory_bytes, filter_size)


def _calc_memory_bytes_paganin_filter_tomopy(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    from httomolibgpu.prep.phase import _shift_bit_length

    # Input (unpadded)
    unpadded_in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize

    # estimate padding size here based on non_slice dimensions
    pad_tup = []
    for dim_len in non_slice_dims_shape:
        diff = _shift_bit_length(dim_len + 1) - dim_len
        if dim_len % 2 == 0:
            pad_width = diff // 2
            pad_width = (pad_width, pad_width)
        else:
            # need an uneven padding for odd-number lengths
            left_pad = diff // 2
            right_pad = diff - left_pad
            pad_width = (left_pad, right_pad)
        pad_tup.append(pad_width)

    # Padded input
    padded_in_slice_size = (
        (non_slice_dims_shape[0] + pad_tup[0][0] + pad_tup[0][1])
        * (non_slice_dims_shape[1] + pad_tup[1][0] + pad_tup[1][1])
        * dtype.itemsize
    )

    # Padded input cast to `complex64`
    complex_slice = padded_in_slice_size / dtype.itemsize * np.complex64().nbytes

    # Plan size for 2D FFT
    ny = non_slice_dims_shape[0] + pad_tup[0][0] + pad_tup[0][1]
    nx = non_slice_dims_shape[1] + pad_tup[1][0] + pad_tup[1][1]
    fftplan_slice_size = cufft_estimate_2d(
        nx=nx,
        ny=ny,
        fft_type=CufftType.CUFFT_C2C,
    )

    # Size of "reciprocal grid" generated, based on padded projections shape
    grid_size = np.prod((ny, nx)) * np.float32().nbytes
    filter_size = grid_size

    # Size of cropped/unpadded + cast to float32 result of 2D IFFT
    cropped_float32_res_slice = np.prod(non_slice_dims_shape) * np.float32().nbytes

    # Size of negative log of cropped float32 result of 2D IFFT
    negative_log_slice = cropped_float32_res_slice

    # If the FFT plan size is negligible for some reason, this changes where the peak GPU
    # memory usage occurs. Hence, the if/else branching below for calculating the total bytes.
    NEGLIGIBLE_FFT_PLAN_SIZE = 16
    if fftplan_slice_size < NEGLIGIBLE_FFT_PLAN_SIZE:
        tot_memory_bytes = int(
            unpadded_in_slice_size + padded_in_slice_size + complex_slice
        )
    else:
        tot_memory_bytes = int(
            unpadded_in_slice_size
            + padded_in_slice_size
            + complex_slice
            # The padded float32 array is deallocated when a copy is made when casting to complex64
            # and the variable `padded_tomo` is reassigned to the complex64 version
            - padded_in_slice_size
            + fftplan_slice_size
            + cropped_float32_res_slice
            + negative_log_slice
        )

    subtract_bytes = int(filter_size + grid_size)

    return (tot_memory_bytes, subtract_bytes)
