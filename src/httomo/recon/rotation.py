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
# Created Date: 27/October/2022
# version ='0.1'
# ---------------------------------------------------------------------------
"""Modules for finding the center of rotation """ 

import numpy as np
from numpy import ndarray
from scipy import stats
import scipy.ndimage as ndi

def find_center_360(data: ndarray,
                    win_width: int = 10,
                    side: set = None,
                    denoise: bool = True,
                    norm: bool = False,
                    use_overlap: bool = False) -> tuple((float,float,int,float)):
    """Find the center-of-rotation (COR) in a 360-degree scan with offset COR use
    the method presented in Ref. [1] by Nghia Vo.

    Parameters
    ----------
    sino_360 : ndarray
        2D array. 360-degree sinogram.
    win_width : int
        Window width used for finding the overlap area.
    side : {None, 0, 1}, optional
        Overlap size. Only there options: None, 0, or 1. "None" corresponding
        to fully automated determination. "0" corresponding to the left side.
        "1" corresponding to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    cor : float
        Center-of-rotation.
    overlap : float
        Width of the overlap area between two halves of the sinogram.
    side : int
        Overlap side between two halves of the sinogram.
    overlap_position : float
        Position of the window in the first image giving the best
        correlation metric.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.418448
    """
    (nrow, ncol) = data.shape
    nrow_180 = nrow // 2 + 1
    sino_top = data[0:nrow_180, :]
    sino_bot = np.fliplr(data[-nrow_180:, :])
    (overlap, side, overlap_position) = _find_overlap(
        sino_top, sino_bot, win_width, side, denoise, norm, use_overlap)
    if side == 0:
        cor = overlap / 2.0 - 1.0
    else:
        cor = ncol - overlap / 2.0 - 1.0
    return cor, overlap, side, overlap_position



def _find_overlap(mat1, mat2, win_width, side=None, denoise=True, norm=False,
                 use_overlap=False):
    """
    Find the overlap area and overlap side between two images (Ref. [1]) where
    the overlap side referring to the first image.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 :  array_like
        2D array. Projection image or sinogram image.
    win_width : int
        Width of the searching window.
    side : {None, 0, 1}, optional
        Only there options: None, 0, or 1. "None" corresponding to fully
        automated determination. "0" corresponding to the left side. "1"
        corresponding to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    overlap : float
        Width of the overlap area between two images.
    side : int
        Overlap side between two images.
    overlap_position : float
        Position of the window in the first image giving the best
        correlation metric.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.418448
    """
    (_, ncol1) = mat1.shape
    (_, ncol2) = mat2.shape
    win_width = np.int16(np.clip(win_width, 6, min(ncol1, ncol2) // 2))
    if side == 1:
        (list_metric, offset) = _search_overlap(mat1, mat2, win_width, side,
                                               denoise, norm, use_overlap)
        (_, overlap_position) = _calculate_curvature(list_metric)
        overlap_position = overlap_position + offset
        overlap = ncol1 - overlap_position + win_width // 2
    elif side == 0:
        (list_metric, offset) = _search_overlap(mat1, mat2, win_width, side,
                                               denoise, norm, use_overlap)
        (_, overlap_position) = _calculate_curvature(list_metric)
        overlap_position = overlap_position + offset
        overlap = overlap_position + win_width // 2
    else:
        (list_metric1, offset1) = _search_overlap(mat1, mat2, win_width, 1,
                                                 norm, denoise, use_overlap)
        (list_metric2, offset2) = _search_overlap(mat1, mat2, win_width, 0,
                                                 norm, denoise, use_overlap)
        (curvature1, overlap_position1) = _calculate_curvature(list_metric1)
        overlap_position1 = overlap_position1 + offset1
        (curvature2, overlap_position2) = _calculate_curvature(list_metric2)
        overlap_position2 = overlap_position2 + offset2
        if curvature1 > curvature2:
            side = 1
            overlap_position = overlap_position1
            overlap = ncol1 - overlap_position + win_width // 2
        else:
            side = 0
            overlap_position = overlap_position2
            overlap = overlap_position + win_width // 2
    return overlap, side, overlap_position

def _search_overlap(mat1, mat2, win_width, side, denoise=True, norm=False,
                   use_overlap=False):
    """
    Calculate the correlation metrics between a rectangular region, defined
    by the window width, on the utmost left/right side of image 2 and the
    same size region in image 1 where the region is slided across image 1.

    Parameters
    ----------
    mat1 : array_like
        2D array. Projection image or sinogram image.
    mat2 : array_like
        2D array. Projection image or sinogram image.
    win_width : int
        Width of the searching window.
    side : {0, 1}
        Only two options: 0 or 1. It is used to indicate the overlap side
        respects to image 1. "0" corresponds to the left side. "1" corresponds
        to the right side.
    denoise : bool, optional
        Apply the Gaussian filter if True.
    norm : bool, optional
        Apply the normalization if True.
    use_overlap : bool, optional
        Use the combination of images in the overlap area for calculating
        correlation coefficients if True.

    Returns
    -------
    list_metric : array_like
        1D array. List of the correlation metrics.
    offset : int
        Initial position of the searching window where the position
        corresponds to the center of the window.
    """
    if denoise is True:
        mat1 = ndi.gaussian_filter(mat1, (2, 2), mode='reflect')
        mat2 = ndi.gaussian_filter(mat2, (2, 2), mode='reflect')
    (nrow1, ncol1) = mat1.shape
    (nrow2, ncol2) = mat2.shape
    if nrow1 != nrow2:
        raise ValueError("Two images are not at the same height!!!")
    win_width = np.int16(np.clip(win_width, 6, min(ncol1, ncol2) // 2 - 1))
    offset = win_width // 2
    win_width = 2 * offset  # Make it even
    ramp_down = np.linspace(1.0, 0.0, win_width)
    ramp_up = 1.0 - ramp_down
    wei_down = np.tile(ramp_down, (nrow1, 1))
    wei_up = np.tile(ramp_up, (nrow1, 1))
    if side == 1:
        mat2_roi = mat2[:, 0:win_width]
        mat2_roi_wei = mat2_roi * wei_up
    else:
        mat2_roi = mat2[:, ncol2 - win_width:]
        mat2_roi_wei = mat2_roi * wei_down
    list_mean2 = np.mean(np.abs(mat2_roi), axis=1)
    list_pos = np.arange(offset, ncol1 - offset)
    num_metric = len(list_pos)
    list_metric = np.ones(num_metric, dtype=np.float32)
    for i, pos in enumerate(list_pos):
        mat1_roi = mat1[:, pos - offset:pos + offset]
        if use_overlap is True:
            if side == 1:
                mat1_roi_wei = mat1_roi * wei_down
            else:
                mat1_roi_wei = mat1_roi * wei_up
        if norm is True:
            list_mean1 = np.mean(np.abs(mat1_roi), axis=1)
            list_fact = list_mean2 / list_mean1
            mat_fact = np.transpose(np.tile(list_fact, (win_width, 1)))
            mat1_roi = mat1_roi * mat_fact
            if use_overlap is True:
                mat1_roi_wei = mat1_roi_wei * mat_fact
        if use_overlap is True:
            mat_comb = mat1_roi_wei + mat2_roi_wei
            list_metric[i] = (_correlation_metric(mat1_roi, mat2_roi)
                              + _correlation_metric(mat1_roi, mat_comb)
                              + _correlation_metric(mat2_roi, mat_comb)) / 3.0
        else:
            list_metric[i] = _correlation_metric(mat1_roi, mat2_roi)
    min_metric = np.min(list_metric)
    if min_metric != 0.0:
        list_metric = list_metric / min_metric
    return list_metric, offset

def _calculate_curvature(list_metric):
    """
    Calculate the curvature of a fitted curve going through the minimum
    value of a metric list.

    Parameters
    ----------
    list_metric : array_like
        1D array. List of metrics.

    Returns
    -------
    curvature : float
        Quadratic coefficient of the parabola fitting.
    min_pos : float
        Position of the minimum value with sub-pixel accuracy.
    """
    radi = 2
    num_metric = len(list_metric)
    min_pos = np.clip(
        np.argmin(list_metric), radi, num_metric - radi - 1)
    list1 = list_metric[min_pos - radi:min_pos + radi + 1]
    (afact1, _, _) = np.polyfit(np.arange(0, 2 * radi + 1), list1, 2)
    list2 = list_metric[min_pos - 1:min_pos + 2]
    (afact2, bfact2, _) = np.polyfit(
        np.arange(min_pos - 1, min_pos + 2), list2, 2)
    curvature = np.abs(afact1)
    if afact2 != 0.0:
        num = - bfact2 / (2 * afact2)
        if (num >= min_pos - 1) and (num <= min_pos + 1):
            min_pos = num
    return curvature, np.float32(min_pos)


def _correlation_metric(mat1, mat2):
    """
    Calculate the correlation metric. Smaller metric corresponds to better
    correlation.

    Parameters
    ---------
    mat1 : array_like
    mat2 : array_like

    Returns
    -------
    float
        Correlation metric.
    """
    metric = np.abs(
        1.0 - stats.pearsonr(mat1.flatten('F'), mat2.flatten('F'))[0])
    return metric