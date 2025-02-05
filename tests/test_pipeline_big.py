import re
import subprocess
from typing import Callable, List, Tuple, Union

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import Image
from plumbum import local
from .conftest import _change_value_parameters_method_pipeline


@pytest.mark.pipebig
def test_pipeline_gpu_FBP_diad_k11_38731(
    get_files: Callable,
    cmd,
    diad_k11_38731,
    gpu_pipeline_diad_FBP_noimagesaving,
    gpu_diad_FBP_k11_38731_npz,
    output_folder,
):
    # NOTE that the intermediate file with file-based processing will be saved to /tmp

    cmd.pop(4)  #: don't save all
    cmd.insert(5, diad_k11_38731)
    cmd.insert(7, gpu_pipeline_diad_FBP_noimagesaving)
    cmd.insert(8, output_folder)
    cmd.insert(9, "--max-memory")
    cmd.insert(10, "40G")
    cmd.insert(11, "--reslice-dir")
    cmd.insert(12, "/scratch/jenkins_agent/workspace/imaging_httomo_full_pipelines/")

    subprocess.check_output(cmd)

    files = get_files("output_dir/")

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    data_gt = gpu_diad_FBP_k11_38731_npz["data"]
    axis_slice = gpu_diad_FBP_k11_38731_npz["axis_slice"]
    (slices, sizeX, sizeY) = np.shape(data_gt)

    step = axis_slice // (slices + 2)
    # store for the result
    data_result = np.zeros((slices, sizeX, sizeY), dtype=np.float32)

    path_to_data = "data/"
    for file_to_open in h5_files:
        if "httomolibgpu-FBP" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                index_prog = step
                for i in range(slices):
                    data_result[i, :, :] = file_to_open[path_to_data][
                        :, index_prog, :
                    ]
                    index_prog += step
        else:
            raise FileNotFoundError("File with httomolibgpu-FBP string cannot be found")

    residual_im = data_gt - data_result
    res_norm = np.linalg.norm(residual_im.flatten()).astype("float32")
    assert res_norm < 1e-6
