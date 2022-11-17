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

def dezinger_filter3d(data: ndarray, 
                      glob_stats: tuple,
                      radius_kernel: int = 1,
                      mu_dezinger: float = 0.1,
                      ncore: int = 1) -> ndarray:
    """Dezinger filter (aka remove_outlier in TomoPy) in 3D from the Larix toolbox (C - implementation)

    Parameters
    ----------
    data : ndarray
        Input array.
    glob_stats: tuple
        Global statistics of input data in a tuple given as: (min, max, mean, std_var).        
    radius_kernel : int, optional
        The radius of the median kernel (e.g.,the full size 3D kernel is (2*radius_kernel+1)^3).
    mu_dezinger : float, optional
        A threshold dezinging parameter, when it equals zero all values are median-filtered
    ncore : int, optional
        The number of CPU cores.

    Returns
    -------
    ndarray
        Dezinger filtered array.
    """
    from larix.methods.misc import MEDIAN_DEZING

    return MEDIAN_DEZING(data, radius_kernel, glob_stats[3]*mu_dezinger, ncore)


def inpainting_filter3d(data: ndarray,
                        mask: ndarray,
                        number_of_iterations: int = 3,
                        windowsize_half: int = 5,
                        method_type: str = "random",
                        ncore: int = 1) -> ndarray:
    """Inpainting filter in 3D from the Larix toolbox (C - implementation). 
    A morphological inpainting scheme which progresses from the edge of the mask inwards. 
    It acts like a diffusion-type process but significantly faster in convergence.

    Parameters
    ----------
    data : ndarray
        Input array.
    mask : ndarray
        Input binary mask (uint8) the same size as data, integer 1 will define the inpainting area.
    number_of_iterations : int, optional
        An additional number of iterations to run after the region has been inpainted (smoothing effect).
    windowsize_half : int, optional
        Half-window size of the searching window (neighbourhood window).
    method_type : str, optional
        method how to select a value in the neighbourhood: mean, meadian or random.
    ncore : int, optional
        The number of CPU cores.

    Returns
    -------
    ndarray
        Inpainted array.
    """
    from larix.methods.misc import INPAINT_EUCL_WEIGHTED
  
    return INPAINT_EUCL_WEIGHTED(data, mask, number_of_iterations, windowsize_half, method_type, ncore)