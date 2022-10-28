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
"""HTTomo Modules for loading/saving images""" 

import numpy as np
from numpy import ndarray
from mpi4py.MPI import Comm
import os
from PIL import Image
import skimage.exposure as exposure

def _save_image(array2d, glob_stats, bits, jpeg_quality, path_to_out_file):    
    """rescale to the bit chosen and save an image

    Args:
        array2d (array): given array
        glob_stats (tuple): statistics for normalisation
        bits (int): chosen bits number
        jpeg_quality (int): chosen quality for jpegs
        path_to_out_file (str): full path to the file
    """
    if bits == 8:
        array2d = exposure.rescale_intensity(array2d, in_range=(glob_stats[0], glob_stats[1]), out_range=(0,255)).astype(np.uint8)
    elif bits == 16:
        array2d = exposure.rescale_intensity(array2d, in_range=(glob_stats[0], glob_stats[1]), out_range=(0,65535)).astype(np.uint16)
    else:
        array2d = exposure.rescale_intensity(array2d, in_range=(glob_stats[0], glob_stats[1]), out_range=(glob_stats[0], glob_stats[1])).astype(np.float32)    
    img = Image.fromarray(array2d)
    img.save(path_to_out_file, quality=jpeg_quality)

def save(data: ndarray,
         out_folder_path: str,
         glob_stats: tuple,  
         comm: Comm,
         axis: int = 0,         
         file_format: str = 'tif',
         bits: int = 8,
         jpeg_quality: int = 95):
    """Saving data as 2D images

    Parameters
    ----------
    data : ndarray
        Required input array.
    out_folder_path : str
        Required path to the output folder (subfolder "images" will be generated). 
    glob_stats: tuple
        Collected global statistics of input data in a tuple given as: (min, max, mean, std_var).
    comm: int
        MPI communicator.
    axis : int, optional
        Specify the axis to use to slice the data (if data is 3D array).
    file_format : str, optional
        Specify file format, e.g. "png", "jpeg" or "tiff".
    bits : int, optional
            Specify the number of bits (8, 16 or 32-bit).
    jpeg_quality : int, optional
            Specify the quality of the jpeg image.
        
    """ 
    if bits not in [8,16,32]:
        bits = 32
        print("The selected bit type %s is not available, resetting to 32 bit \n" % str(bits))
    # create the output folder
    path_to_images_dir = out_folder_path + '/' + "images" + str(bits) + "bit" + "_" + str(file_format) + '/'    
    if not os.path.exists(path_to_images_dir):
        os.makedirs(path_to_images_dir)    
    
    data_full_shape = np.shape(data)    
    if data.ndim == 3:
        slice_dim_size=data_full_shape[axis]
        for i in range(slice_dim_size):
            filename = '%s%05i.%s' % (path_to_images_dir, i + comm.rank*slice_dim_size, file_format)
            _save_image(data.take(indices=i,axis=axis), glob_stats, bits, jpeg_quality, filename)
    else:
        _save_image(data, glob_stats, bits, jpeg_quality, '%s%05i.%s' % (path_to_images_dir, 1, file_format))
        
    print(comm.rank)
    return