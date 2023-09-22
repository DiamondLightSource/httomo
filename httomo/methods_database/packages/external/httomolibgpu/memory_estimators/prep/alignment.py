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
"""Modules for data correction (e.g. distortion correction)"""

import math
from typing import Tuple
import numpy as np

__all__ = [
    "_calc_memory_bytes_distortion_correction_proj_discorpy",
]

def _calc_memory_bytes_distortion_correction_proj_discorpy(
        non_slice_dims_shape: Tuple[int, int],
        dtype: np.dtype,
        **kwargs,
) -> Tuple[int, int]:
    # calculating memory is a bit more involved as various small temporary arrays
    # are used. We revert to a rough estimation using only the larger elements and
    # a safety margin
    height, width = non_slice_dims_shape[0], non_slice_dims_shape[1]
    lists_size = (width + height) * np.float64().nbytes
    meshgrid_size = (width * height * 2) * np.float64().nbytes
    ru_mat_size = meshgrid_size // 2
    fact_mat_size = ru_mat_size
    xd_mat_size = yd_mat_size = fact_mat_size // 2   # float32
    indices_size = xd_mat_size + yd_mat_size

    slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    processing_size = slice_size * 4  # temporaries in final for loop
    
    subtract_bytes = 0
    subtract_bytes += (
        lists_size
        + meshgrid_size
        + ru_mat_size
        + fact_mat_size * 2
        + xd_mat_size
        + yd_mat_size
        + processing_size
        + indices_size * 3  # The x 3 here is for additional safety margin 
    )                       # to allow for memory for temporaries
    return (1, subtract_bytes)