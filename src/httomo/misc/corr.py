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
from mpi4py.MPI import Comm

from httomo._stats.globals import max_min_mean_std


def _stats_global_calc(data, flats, darks, mu_dezinger, comm):
    """global stats pre-calculation
    """
    if mu_dezinger is not None and mu_dezinger > 0.0 :
        maxval, minval, mean, std_var_data = max_min_mean_std(data, comm=comm) # for data 
        maxval, minval, mean, std_var_flats = max_min_mean_std(flats, comm=comm) # for flats
        maxval, minval, mean, std_var_darks = max_min_mean_std(darks, comm=comm) # for darks
        std_all_data = (mu_dezinger*std_var_data, mu_dezinger*std_var_flats, mu_dezinger*std_var_darks)
    else:
        std_all_data = (1, 1, 1)
    return std_all_data

def median_filter3d(data, radius_kernel=1, global_std=1, ncore=1):
    """Median filter in 3D from the Larix toolbox (C - implementation)

    Parameters
    ----------
    data : ndarray
        Input array.
    radius_kernel : int, optional
        The radius of the median kernel (the full size 3D kernel is (2*radius_kernel+1)^3).
    global_std : float, optional
        The standard deviation of the input dataset, calculated automatically in httomo.
    ncore : int, optional
        The number of CPU cores.

    Returns
    -------
    ndarray
        Median filtered 3D array.
    """
    from larix.methods.misc import MEDIAN_FILT
    
    # global stats pre-calculation 
    std_all_data = _stats_global_calc(data, flats, darks, mu_dezinger, comm)
        
    if mu_dezinger > 0.0:
        data = MEDIAN_DEZING(data, radius_kernel, std_all_data[0], ncores)
    else:
        data = MEDIAN_FILT(data, radius_kernel, ncores)
    if flats is not None:
        if mu_dezinger > 0.0:
            flats = MEDIAN_DEZING(flats, radius_kernel, std_all_data[1], ncores)            
        else:
            flats = MEDIAN_FILT(flats, radius_kernel, ncores)            
    if darks is not None:
        if mu_dezinger > 0.0:
            darks = MEDIAN_DEZING(darks, radius_kernel, std_all_data[2], ncores)            
        else:            
            darks = MEDIAN_FILT(darks, radius_kernel, ncores)

    return data, flats, darks



def median_filter(data: ndarray, method_name: str, flats: ndarray, darks: ndarray,
          radius_kernel: int, mu_dezinger: float, ncores: int, comm: Comm):    
    """Median filter in 3D from the Larix toolbox (C - implementation)

    Parameters
    ----------
    data : ndarray
        Input array.
    size : int, optional
        The size of the filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.

    Returns
    -------
    ndarray
        Median filtered 3D array.
    """
    
    """Wrapper for the median and dezinger filtration methods in the larix
    library.

    Args:
        data: A numpy array containing raw projections.
        flats: A numpy array containing the flatfield projections.
        darks: A numpy array containing the dark projections.        
        radius_kernel: a radius of the median kernel (the full size 3D kernel is (2*radius_kernel+1)^3)       
        mu_dezinger: a dezinging parameter, when 0.0 - median filter, when > 0.0 - dezinger
        ncores: The number of CPU cores per process
        comm: The MPI communicator to use.

    Returns:
        tuple[ndarray, ndarray, ndarray]: A tuple of numpy arrays containing the
            filtered projections, flatfields and darkfields.
    """
    from larix.methods.misc import MEDIAN_FILT, MEDIAN_DEZING
    
    # global stats pre-calculation 
    std_all_data = _stats_global_calc(data, flats, darks, mu_dezinger, comm)
        
    if mu_dezinger > 0.0:
        data = MEDIAN_DEZING(data, radius_kernel, std_all_data[0], ncores)
    else:
        data = MEDIAN_FILT(data, radius_kernel, ncores)
    if flats is not None:
        if mu_dezinger > 0.0:
            flats = MEDIAN_DEZING(flats, radius_kernel, std_all_data[1], ncores)            
        else:
            flats = MEDIAN_FILT(flats, radius_kernel, ncores)            
    if darks is not None:
        if mu_dezinger > 0.0:
            darks = MEDIAN_DEZING(darks, radius_kernel, std_all_data[2], ncores)            
        else:            
            darks = MEDIAN_FILT(darks, radius_kernel, ncores)

    return data, flats, darks
