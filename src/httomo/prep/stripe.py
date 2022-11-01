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
# Created By  : Daniil Kazantsev <scientificsoftware@diamond.ac.uk>
# Created Date: 01/november/2022
# version ='0.1'
# ---------------------------------------------------------------------------
"""Modules for stripes detection and removal""" 

import numpy as np
from numpy import ndarray

def _gradient(data, axis):
    return np.gradient(data, axis=axis)

def detect_stripes(data: ndarray,
                   threshold_val: float = 0.1,
                   ncore: int = 1) -> ndarray:
    """A module to detect stripes in the data

    Parameters
    ----------
    data : ndarray
        Input array.
    radius_kernel : int, optional
        The radius of the median kernel (e.g., the full size 3D kernel is (2*radius_kernel+1)^3).
    ncore : int, optional
        The number of CPU cores.

    Returns
    -------
    ndarray
        Median filtered array.
    """
    from larix.methods.misc import MEDIAN_FILT
    
    return MEDIAN_FILT(data, radius_kernel, ncore)
