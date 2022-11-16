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
# Created Date: 25/October/2022
# version ='0.1'
# ---------------------------------------------------------------------------
"""HTTomo Modules for data segmentation and thresholding""" 

import numpy as np
from numpy import ndarray

def otsu_thresholding(data: ndarray,
                      foreground: bool = True) -> float:
    """Otsu thresholding of a 2D image

    Parameters
    ----------
    data : ndarray
        Input array.
    foreground : bool, optional
        data is thresholded to get the foreground, otherwise background

    Returns
    -------
    ndarray
        A 2D thresholded image.
    """
    from skimage import filters
    
    val = filters.threshold_otsu(data)

    return val