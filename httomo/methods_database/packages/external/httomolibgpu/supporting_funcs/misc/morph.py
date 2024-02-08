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

__all__ = [
    "_calc_memory_bytes_data_resampler",
]


def _calc_memory_bytes_data_resampler(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    newshape = kwargs["newshape"]
    method = kwargs["method"]

    input_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    xi = 2 * np.prod(newshape) * dtype.itemsize
    output_size = np.prod(newshape) * dtype.itemsize

    # interpolation happens in 2d so we should allocate for it, the exact value is unknown
    if method == "nearest":
        interpolator = 3 * (input_size + output_size)
    if method == "linear":
        interpolator = 4 * (input_size + output_size)

    tot_memory_bytes = input_size + output_size + interpolator
    return (tot_memory_bytes, xi)
