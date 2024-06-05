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
"""Modules for memory estimation for reconstruction algorithms"""

import math
from typing import Tuple
import numpy as np
from httomo.cufft import CufftType, cufft_estimate_1d

__all__ = [
    "_calc_memory_bytes_FBP",
    "_calc_memory_bytes_SIRT",
    "_calc_memory_bytes_CGLS",
    "_calc_output_dim_FBP",
    "_calc_output_dim_SIRT",
    "_calc_output_dim_CGLS",
]


def __calc_output_dim_recon(non_slice_dims_shape, **kwargs):
    """Function to calculate output dimensions for all reconstructors.
    The change of the dimension depends either on the user-provided "recon_size"
    parameter or taken as the size of the horizontal detector (default).

    """
    DetectorsLengthH = non_slice_dims_shape[1]
    recon_size = kwargs["recon_size"]
    if recon_size is None:
        recon_size = DetectorsLengthH
    output_dims = (recon_size, recon_size)
    return output_dims


def _calc_output_dim_FBP(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_SIRT(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_output_dim_CGLS(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)


def _calc_memory_bytes_FBP(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    det_width = non_slice_dims_shape[1]
    output_dims = _calc_output_dim_FBP(non_slice_dims_shape, **kwargs)

    input_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    filtered_input_slice_size = np.prod(non_slice_dims_shape) * np.float32().itemsize
    # astra backprojection will generate an output array
    recon_output_size = np.prod(output_dims) * np.float32().itemsize

    # filter needs to be created once and substracted from the total memory
    filter_size = (det_width // 2 + 1) * np.float32().itemsize

    # this stores the result of applying FFT to data. proj_f array in the code.
    filtered_freq_slice = (
        non_slice_dims_shape[0] * (det_width // 2 + 1) * np.complex64().itemsize
    )

    batch = non_slice_dims_shape[0]
    SLICES = 200  # dummy multiplier+divisor to pass large batch size threshold
    fftplan_size = (
        cufft_estimate_1d(
            nx=det_width,
            fft_type=CufftType.CUFFT_R2C,
            batch=batch * SLICES,
        )
        / SLICES
    )
    ifftplan_size = (
        cufft_estimate_1d(
            nx=det_width,
            fft_type=CufftType.CUFFT_C2R,
            batch=batch * SLICES,
        )
        / SLICES
    )

    tot_memory_bytes = int(
        input_slice_size
        + filtered_input_slice_size
        + filtered_freq_slice
        + fftplan_size
        + ifftplan_size
        + recon_output_size
    )

    # Backprojection ASTRA part estimations
    tot_memory_ASTRA_BP_bytes = 5 * input_slice_size + recon_output_size
    if tot_memory_ASTRA_BP_bytes > tot_memory_bytes:
        tot_memory_bytes = tot_memory_ASTRA_BP_bytes
    return (tot_memory_bytes, filter_size)


def _calc_memory_bytes_SIRT(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    DetectorsLengthH = non_slice_dims_shape[1]
    # calculate the output shape
    output_dims = _calc_output_dim_SIRT(non_slice_dims_shape, **kwargs)

    in_data_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_data_size = np.prod(output_dims) * dtype.itemsize

    astra_projection = 2.5 * (in_data_size + out_data_size)

    tot_memory_bytes = 2 * in_data_size + 2 * out_data_size + astra_projection
    return (tot_memory_bytes, 0)


def _calc_memory_bytes_CGLS(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    DetectorsLengthH = non_slice_dims_shape[1]
    # calculate the output shape
    output_dims = _calc_output_dim_CGLS(non_slice_dims_shape, **kwargs)

    in_data_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_data_size = np.prod(output_dims) * dtype.itemsize

    astra_projection = 2.5 * (in_data_size + out_data_size)

    tot_memory_bytes = 2 * in_data_size + 2 * out_data_size + astra_projection
    return (tot_memory_bytes, 0)
