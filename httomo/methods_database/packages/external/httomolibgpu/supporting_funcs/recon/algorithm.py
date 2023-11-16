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
    "_calc_output_dim_FBP",
]

def __calc_output_dim_recon(non_slice_dims_shape, **kwargs):
    """Function to calculate output dimensions for all reconstructors. 
    The change of the dimension depends either on the user-provided "recon_size"
    parameter or taken as the size of the horizontal detector (default).

    """
    DetectorsLengthH = non_slice_dims_shape[1]
    recon_size = kwargs['recon_size']
    if recon_size is None:
        recon_size = DetectorsLengthH
    output_dims = (recon_size, recon_size)
    return output_dims

def _calc_output_dim_FBP(non_slice_dims_shape, **kwargs):
    return __calc_output_dim_recon(non_slice_dims_shape, **kwargs)

def _calc_memory_bytes_FBP(
        non_slice_dims_shape: Tuple[int, int],
        dtype: np.dtype,
        **kwargs,
) -> Tuple[int, int]:    
    det_width = non_slice_dims_shape[1]
    output_dims = _calc_output_dim_FBP(non_slice_dims_shape,  **kwargs)
    
    in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    filter_size = (det_width//2+1) * np.float32().itemsize

    batch = non_slice_dims_shape[0]
    SLICES = 200 # dummy multiplier+divisor to pass large batch size threshold
    fftplan_size = cufft_estimate_1d(
        nx=det_width,
        fft_type=CufftType.CUFFT_R2C,
        batch=batch*SLICES,
    ) / SLICES
    ifftplan_size = cufft_estimate_1d(
        nx=det_width,
        fft_type=CufftType.CUFFT_C2R,
        batch=batch*SLICES,
    ) / SLICES

    # astra backprojection will generate an output array 
    astra_out_size = (np.prod(output_dims) * np.float32().itemsize)
    tot_memory_bytes = int(
        2*in_slice_size +
        fftplan_size +
        ifftplan_size +
        2.5*astra_out_size
    )
    return (tot_memory_bytes, filter_size)
