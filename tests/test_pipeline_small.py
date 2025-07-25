import re
import subprocess
from typing import Callable, List, Tuple, Union

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import Image
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
    assert (
        "Using GPU 0 to transfer data of shape (180, 128, 160)" in verbose_log_contents
    )


# TODO: rewrite and move to test_pipeline_big
# @pytest.mark.small_data
# def test_tomo_standard_testing_pipeline_output_with_save_all(
#     get_files: Callable,
#     cmd,
#     standard_data,
#     standard_loader,
#     testing_pipeline,
#     output_folder,
#     merge_yamls,
# ):
#     cmd.insert(7, standard_data)
#     merge_yamls(standard_loader, testing_pipeline)
#     cmd.insert(8, "temp.yaml")
#     cmd.insert(9, output_folder)
#     subprocess.check_output(cmd)

#     files = get_files("output_dir/")
#     assert len(files) == 11

#     _check_yaml(files, "temp.yaml")
#     _check_tif(files, 3, (160, 160))

#     #: check the generated h5 files
#     h5_files = list(filter(lambda x: ".h5" in x, files))
#     assert len(h5_files) == 5

#     for file_to_open in h5_files:
#         if "tomopy-recon-tomo-gridrec.h5" in file_to_open:
#             with h5py.File(file_to_open, "r") as f:
#                 assert f["data"].shape == (160, 3, 160)
#                 assert f["data"].dtype == np.float32
#                 assert_allclose(np.mean(f["data"]), 0.0015362317, atol=1e-6, rtol=1e-6)
#                 assert_allclose(np.sum(f["data"]), 117.9826, atol=1e-6, rtol=1e-6)


# TODO: we will be testing this in test_pipeline_big
# def test_i12_testing_pipeline_output(
#     get_files: Callable,
#     cmd,
#     i12_data,
#     i12_loader,
#     testing_pipeline,
#     output_folder,
#     merge_yamls,
# ):
#     cmd.insert(7, i12_data)
#     merge_yamls(i12_loader, testing_pipeline)
#     cmd.insert(8, "temp.yaml")
#     cmd.insert(9, output_folder)
#     subprocess.check_output(cmd)

#     files = get_files("output_dir/")
#     assert len(files) == 18

#     _check_yaml(files, "temp.yaml")

#     log_files = list(filter(lambda x: ".log" in x, files))
#     assert len(log_files) == 2

#     tif_files = list(filter(lambda x: ".tif" in x, files))
#     assert len(tif_files) == 10

#     h5_files = list(filter(lambda x: ".h5" in x, files))
#     assert len(h5_files) == 5

#     gridrec_recon = list(filter(lambda x: "recon-gridrec.h5" in x, h5_files))[0]
#     minus_log_tomo = list(filter(lambda x: "minus_log.h5" in x, h5_files))[0]
#     remove_stripe_fw_tomo = list(
#         filter(lambda x: "remove_stripe_fw.h5" in x, h5_files)
#     )[0]
#     normalize_tomo = list(filter(lambda x: "normalize.h5" in x, h5_files))[0]

#     with h5py.File(gridrec_recon, "r") as f:
#         assert f["data"].shape == (192, 10, 192)
#         assert_allclose(np.sum(f["data"]), 2157.03, atol=1e-2, rtol=1e-6)
#         assert_allclose(np.mean(f["data"]), 0.0058513316, atol=1e-6, rtol=1e-6)
#     with h5py.File(minus_log_tomo, "r") as f:
#         assert_allclose(np.sum(f["data"]), 1756628.4, atol=1e-6, rtol=1e-6)
#         assert_allclose(np.mean(f["data"]), 1.2636887, atol=1e-6, rtol=1e-6)
#     with h5py.File(remove_stripe_fw_tomo, "r") as f:
#         assert_allclose(np.sum(f["data"]), 1766357.8, atol=1e-6, rtol=1e-6)
#         assert_allclose(np.mean(f["data"]), 1.2706878, atol=1e-6, rtol=1e-6)
#     with h5py.File(normalize_tomo, "r") as f:
#         assert f["data"].shape == (724, 10, 192)
#         assert_allclose(np.sum(f["data"]), 393510.72, atol=1e-6, rtol=1e-6)
#         assert_allclose(np.mean(f["data"]), 0.28308493, atol=1e-6, rtol=1e-6)

#     concise_log_file = list(filter(lambda x: "user.log" in x, files))
#     concise_log_contents = _get_log_contents(concise_log_file[0])
#     verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
#     verbose_log_contents = _get_log_contents(verbose_log_file[0])

#     assert "The center of rotation is 95.5" in concise_log_contents
#     assert "The full dataset shape is (724, 10, 192)" in verbose_log_contents
#     assert (
#         "Loading data: tests/test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
#         in verbose_log_contents
#     )
#     assert "Path to data: /1-TempPlugin-tomo/data" in verbose_log_contents
#     assert "Preview: (0:724, 0:10, 0:192)" in verbose_log_contents


# TODO: Will be added to big-data tests when the sample data with separate darks and flats will be available.
# @pytest.mark.small_data
# def test_i12_testing_ignore_darks_flats_pipeline_output(
#     get_files: Callable,
#     cmd,
#     i12_data,
#     i12_loader_ignore_darks_flats,
#     testing_pipeline,
#     output_folder,
#     merge_yamls,
# ):
#     cmd.insert(7, i12_data)
#     merge_yamls(i12_loader_ignore_darks_flats, testing_pipeline)
#     cmd.insert(8, "temp.yaml")
#     cmd.insert(9, output_folder)
#     subprocess.check_output(cmd)

#     files = get_files("output_dir/")
#     assert len(files) == 16

#     _check_yaml(files, "temp.yaml")

#     log_files = list(filter(lambda x: ".log" in x, files))
#     assert len(log_files) == 1

#     tif_files = list(filter(lambda x: ".tif" in x, files))
#     assert len(tif_files) == 10

#     h5_files = list(filter(lambda x: ".h5" in x, files))
#     assert len(h5_files) == 4

#     log_contents = _get_log_contents(log_files[0])
#     assert "The full dataset shape is (724, 10, 192)" in log_contents
#     assert (
#         "Loading data: tests/test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
#         in log_contents
#     )
#     assert "Path to data: /1-TempPlugin-tomo/data" in log_contents
#     assert "Preview: (0:724, 0:10, 0:192)" in log_contents
#     assert (
#         "Running save_task_1 (pattern=projection): save_intermediate_data..."
#         in log_contents
#     )
#     assert (
#         "Running save_task_2 (pattern=projection): save_intermediate_data..."
#         in log_contents
#     )
#     assert (
#         "Running save_task_4 (pattern=sinogram): save_intermediate_data..."
#         in log_contents
#     )
#     assert "The center of rotation for sinogram is 95.5" in log_contents
#     assert (
#         "Running save_task_5 (pattern=sinogram): save_intermediate_data..."
#         in log_contents
#     )
