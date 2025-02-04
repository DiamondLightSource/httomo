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
    gpu_pipeline_diad_FBP,
    gpu_diad_FBP_k11_38731_npz,
    output_folder,
):
    # NOTE that the intermediate file with file-based processing will be saved to /tmp

    cmd.pop(4)  #: don't save all
    cmd.insert(5, diad_k11_38731)
    cmd.insert(6, "--max-slices 50G")
    cmd.insert(7, gpu_pipeline_diad_FBP)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    files = get_files("output_dir/")

    #: check the generated reconstruction (hdf5 file)
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # load the pre-saved numpy array for comparison bellow
    projdata_gt = gpu_diad_FBP_k11_38731_npz["projdata"]
    (slices, dety, detx) = np.shape(projdata_gt)
    step = detx // (slices + 2)

    # store for the result
    projdata_result = np.empty((slices, dety, detx))

    path_to_data = "data/"
    for file_to_open in h5_files:
        if "httomolibgpu-FBP-tomo.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                index_prog = step
                for i in range(slices):
                    projdata_result[i, :, :] = file_to_open[path_to_data][
                        index_prog, :, :
                    ]
                    index_prog += step

    res_norm = np.linalg.norm((projdata_gt - projdata_result)).astype("float32")
    assert res_norm < 1e-6
