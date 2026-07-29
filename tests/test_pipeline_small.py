import re
import subprocess
from typing import Callable, List, Tuple, Union

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from plumbum import local
from .conftest import change_value_parameters_method_pipeline, check_tif

PATTERN = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")


def _get_log_contents(file):
    """return contents of the user.log file"""
    with open(file, "r") as f:
        log_contents = f.read()

    #: check that the generated log file has no ansi escape sequence
    # assert not PATTERN.search(log_contents)

    return log_contents


@pytest.mark.small_data
def test_run_pipeline_tomopy_gridrec(
    get_files: Callable, cmd, standard_data, tomopy_gridrec, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, tomopy_gridrec)
    cmd.insert(8, output_folder)

    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files(output_folder)
    assert len(files) == 132  # 128 images + yaml, 2 logfiles, intermediate

    check_tif(files, 128, (160, 160))

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2

    concise_log_file = list(filter(lambda x: "user.log" in x, files))
    concise_log_contents = _get_log_contents(concise_log_file[0])
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert f"{concise_log_file[0]}" in concise_log_contents
    assert "The center of rotation is 79.5" in concise_log_contents
    assert "The full dataset shape is (220, 128, 160)" in verbose_log_contents
    assert "Loading data: tests/test_data/tomo_standard.nxs" in verbose_log_contents
    assert "Path to data: /entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 0:128, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 128, 160) of type uint16" in verbose_log_contents


@pytest.mark.small_data
def test_run_pipeline_FBP3d_tomobar(
    get_files: Callable, cmd, standard_data, FBP3d_tomobar, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, FBP3d_tomobar)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files(output_folder)
    assert len(files) == 132

    check_tif(files, 128, (160, 160))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    for file_to_open in h5_files:
        if "FBP3d_tomobar" in file_to_open:
            h5f = h5py.File(file_to_open, "r")
            assert h5f["data"].shape == (160, 128, 160)
            assert h5f["data"].dtype == np.float32
            h5f.close()
        else:
            raise FileNotFoundError("File with FBP3d_tomobar string cannot be found")

    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    user_log_file = list(filter(lambda x: "user.log" in x, files))
    assert len(verbose_log_file) == 1
    assert len(user_log_file) == 1
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert f"{user_log_file[0]}" in verbose_log_contents
    assert "The full dataset shape is (220, 128, 160)" in verbose_log_contents
    assert "Loading data: tests/test_data/tomo_standard.nxs" in verbose_log_contents
    assert "Path to data: /entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 0:128, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 128, 160) of type uint16" in verbose_log_contents
    assert "The amount of the available GPU memory is" in verbose_log_contents
    assert (
        "Using GPU 0 to transfer data of shape (180, 128, 160)" in verbose_log_contents
    )


@pytest.mark.small_data
def test_run_pipeline_FBP3d_tomobar_denoising(
    get_files: Callable,
    cmd,
    standard_data,
    FBP3d_tomobar_denoising,
    output_folder,
):
    change_value_parameters_method_pipeline(
        FBP3d_tomobar_denoising,
        method=[
            "find_center_vo",
            "find_center_vo",
            "find_center_vo",
            "total_variation_PD",
        ],
        key=["cor_initialisation_value", "smin", "smax", "iterations"],
        value=[80.0, -20, 20, 200],
    )

    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, FBP3d_tomobar_denoising)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files(output_folder)
    assert len(files) == 132

    check_tif(files, 128, (160, 160))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    for file_to_open in h5_files:
        if "FBP3d_tomobar" in file_to_open:
            h5f = h5py.File(file_to_open, "r")
            assert h5f["data"].shape == (160, 128, 160)
            assert h5f["data"].dtype == np.float32
            h5f.close()
        else:
            raise FileNotFoundError("File with FBP3d_tomobar string cannot be found")

    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    user_log_file = list(filter(lambda x: "user.log" in x, files))
    assert len(verbose_log_file) == 1
    assert len(user_log_file) == 1
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert f"{user_log_file[0]}" in verbose_log_contents
    assert "The full dataset shape is (220, 128, 160)" in verbose_log_contents
    assert "Loading data: tests/test_data/tomo_standard.nxs" in verbose_log_contents
    assert "Path to data: /entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 0:128, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 128, 160) of type uint16" in verbose_log_contents
    assert "The amount of the available GPU memory is" in verbose_log_contents
    assert "Using GPU 0 to transfer data of shape" in verbose_log_contents
