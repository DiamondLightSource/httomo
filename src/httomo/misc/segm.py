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
from skimage import filters

def _get_mask(data, mask, val_intensity, otsu, foreground):
    """gets data binary segmented into a mask  
    
    """
    if otsu:
        # get the intensity value based on Otsu
        val_intensity = filters.threshold_otsu(data)
    if foreground:
        mask[data > val_intensity] = 1
    else:
        mask[data <= val_intensity] = 1
    return mask


def binary_thresholding(data: ndarray,
                        val_intensity: float = 0.1,
                        otsu: bool = False,
                        foreground: bool = True,
                        axis: int = 1) -> ndarray:
    """Performs a binary thresholding of the input data 

    Parameters
    ----------
    data : ndarray
        Input array.
    val_intensity: float
        The grayscale intensity value that defines the binary threshold.      
    otsu: str, optional
        If set to True, val_intensity will be overwritten by Otsu method.
    foreground : bool, optional
        get the foreground, otherwise background.
    axis : int, optional
        Specify the axis to use to slice the data (if data is the 3D array).        

    Returns
    -------
    ndarray
        A binary mask of the input data.
    """

    # initialising output mask
    mask = np.uint8(np.zeros(np.shape(data)))

    data_full_shape = np.shape(data)
    if data.ndim == 3:
        slice_dim_size=data_full_shape[axis]
        for i in range(slice_dim_size):
            _get_mask(data, mask, val_intensity, otsu, foreground)
    else:
        _get_mask(data, mask, val_intensity, otsu, foreground)
    return mask    