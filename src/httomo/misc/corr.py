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
# Created Date: 21/October/2022
# version ='0.1'
# ---------------------------------------------------------------------------
"""Modules for data correction""" 

import numpy as np
from numpy import ndarray

def median_filter3d(data: ndarray,
                    radius_kernel: int = 1,
                    ncore: int = 1) -> ndarray:
    """Median filter in 3D from the Larix toolbox (C - implementation)

    Parameters
    ----------
    data : ndarray
        Input array.
    radius_kernel : int, optional
        The radius of the median kernel (the full size 3D kernel is (2*radius_kernel+1)^3).
    ncore : int, optional
        The number of CPU cores.

    Returns
    -------
    ndarray
        Median filtered 3D array.
    """
    from larix.methods.misc import MEDIAN_FILT
    
    return MEDIAN_FILT(data, radius_kernel, ncore)

def dezinger_filter3d(data: ndarray, 
                      radius_kernel: int = 1,
                      global_std: float = 1.0,
                      mu_dezinger: float = 0.1, 
                      ncore: int = 1) -> ndarray:
    """Dezinger filter (aka remove_outlier in TomoPy) in 3D from the Larix toolbox (C - implementation)

    Parameters
    ----------
    data : ndarray
        Input array.
    radius_kernel : int, optional
        The radius of the median kernel (the full size 3D kernel is (2*radius_kernel+1)^3).
    global_std : float, optional
        The standard deviation of the input dataset, calculated automatically in httomo.
    mu_dezinger : float, optional
        A threshold dezinging parameter, when it equals zero all values are median-filtered
    ncore : int, optional
        The number of CPU cores.

    Returns
    -------
    ndarray
        Dezinger filtered 3D array.
    """
    from larix.methods.misc import MEDIAN_DEZING

    return MEDIAN_DEZING(data, radius_kernel, global_std*mu_dezinger, ncore)