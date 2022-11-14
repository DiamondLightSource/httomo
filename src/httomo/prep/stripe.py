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
                   search_window_dims: tuple = (1,9,1),
                   vert_window_size: float = 5,
                   gradient_gap: int = 2,
                   ncore: int = 1) -> ndarray: 
    """module to detect stripes in sinograms (2D) OR projection data (3D). 
    1. Taking first derrivative of the input in the direction orthogonal to stripes.
    2. Slide horizontal rectangular window orthogonal to stripes direction to accenuate outliers (stripes) using median.
    3. Slide the vertical thin (1 pixel) window to calculate a mean (further accenuates stripes).

    Args:
        data (ndarray): sinogram (2D) [angles x detectorsX] OR projection data (3D) [angles x detectorsY x detectorsX]
        search_window_dims (tuple, optional): searching rectangular window for weights calculation, 
        of a size (detectors_window_height, detectors_window_width, angles_window_depth). Defaults to (1,9,1).
        vert_window_size (float, optional): the half size of the vertical 1D window to calculate mean. Given in percents relative to the size of the angle dimension. Defaults to 5.
        gradient_gap (int, optional):  the gap in pixels with the neighbour while calculating a gradient (1 is the normal gradient). Defaults to 2.
        ncore (int, optional): _description_. Defaults to 1.

    Returns:
        ndarray: The associated weights (needed for thresholding)
    """       
    from larix.methods.misc import STRIPES_DETECT
    
    # calculate weights for stripes
    (stripe_weights, stats_vec) = STRIPES_DETECT(data, search_window_dims, vert_window_size, gradient_gap, ncore)
       
    return stripe_weights


def merge_stripes(data: ndarray,
                   stripe_width_max_perc: float = 5,
                   mask_dilate: int = 2,
                   threshold_stripes: float = 0.1,
                   ncore: int = 1) -> ndarray:
    """module to threshold the obtained stripe weights in sinograms (2D) OR projection data (3D) and merge stripes that are close to each other. 
    
    Args:
        data (ndarray): weigths for sinogram (2D) [angles x detectorsX] OR projection data (3D) [angles x detectorsY x detectorsX]
        stripe_width_max_perc (float, optional):  the maximum width of stripes in the data, given in percents relative to the size of the DetectorX. Defaults to 5.
        mask_dilate (int, optional): the number of pixels/voxels to dilate the obtained mask. Defaults to 2.
        threshold_stripes (float, optional): Threshold the obtained weights to get a binary mask, larger vaules are more sensitive to stripes. Defaults to 0.1.
        ncore (int, optional): _description_. Defaults to 1.

    Returns:
        ndarray: mask_stripe
    """       
    from larix.methods.misc import STRIPES_MERGE
    
    gradientX = _gradient(data, 2)    
    med_val = np.median(np.abs(gradientX).flatten(), axis=0)
    
    # we get a local stats here, needs to be adopted for global stats
    mask_stripe = np.zeros_like(data,dtype="uint8")    
    mask_stripe[data > med_val/threshold_stripes] = 1
    
    # merge stripes that are close to each other
    mask_stripe_merged = STRIPES_MERGE(mask_stripe, stripe_width_max_perc, mask_dilate, ncore)
    
    return mask_stripe_merged