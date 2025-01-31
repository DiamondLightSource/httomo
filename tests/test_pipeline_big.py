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
    get_files: Callable, cmd, diad_k11_38731, gpu_pipeline_diad_FBP, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, diad_k11_38731)
    cmd.insert(7, gpu_pipeline_diad_FBP)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    # assert len(files) == 132

    # _check_tif(files, 128, (160, 160))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    # for file_to_open in h5_files:
    #     if "httomolibgpu-FBP-tomo.h5" in file_to_open:
    #         with h5py.File(file_to_open, "r") as f:
    #             assert f["data"].shape == (160, 128, 160)
    #             assert f["data"].dtype == np.float32

    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    user_log_file = list(filter(lambda x: "user.log" in x, files))
    assert len(verbose_log_file) == 1
    assert len(user_log_file) == 1
    # verbose_log_contents = _get_log_contents(verbose_log_file[0])

    # assert f"{user_log_file[0]}" in verbose_log_contents
    # assert "The full dataset shape is (220, 128, 160)" in verbose_log_contents
    # assert "Loading data: tests/test_data/tomo_standard.nxs" in verbose_log_contents
    # assert "Path to data: /entry1/tomo_entry/data/data" in verbose_log_contents
    # assert "Preview: (0:180, 0:128, 0:160)" in verbose_log_contents
    # assert "Data shape is (180, 128, 160) of type uint16" in verbose_log_contents
    # assert "The amount of the available GPU memory is" in verbose_log_contents
    # assert (
    #     "Using GPU 0 to transfer data of shape (180, 128, 160)" in verbose_log_contents
    # )
