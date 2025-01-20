import re
import subprocess
from typing import Callable, List, Tuple

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import Image
from plumbum import local

PATTERN = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")


def _get_log_contents(file):
    """return contents of the user.log file"""
    with open(file, "r") as f:
        log_contents = f.read()

    #: check that the generated log file has no ansi escape sequence
    # assert not PATTERN.search(log_contents)

    return log_contents


def _compare_two_yamls(original_yaml, copied_yaml):
    with open(original_yaml, "r") as oy, open(copied_yaml, "r") as cy:
        return oy.read() == cy.read()


def _check_yaml(files: List, input_yaml: str):
    # check that the contents of the copied YAML in the output directory matches
    # the contents of the input YAML
    copied_yaml_path = list(filter(lambda x: ".yaml" in x, files)).pop()
    assert _compare_two_yamls(input_yaml, copied_yaml_path)


def _check_tif(files: List, number: int, shape: Tuple):
    # check the .tif files
    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == number

    # check that the image size is correct
    imarray = np.array(Image.open(tif_files[0]))
    assert imarray.shape == shape


def test_tomo_standard_testing_pipeline_output(
    get_files: Callable,
    cmd,
    standard_data,
    standard_loader,
    testing_pipeline,
    output_folder,
    merge_yamls,
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(7, "temp.yaml")
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 7

    _check_yaml(files, "temp.yaml")
    _check_tif(files, 3, (160, 160))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    for file_to_open in h5_files:
        if "tomopy-recon-tomo-gridrec.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (160, 3, 160)
                assert f["data"].dtype == np.float32
                assert_allclose(np.mean(f["data"]), 0.0015362317, atol=1e-6, rtol=1e-6)
                assert_allclose(np.sum(f["data"]), 117.9826, atol=1e-6, rtol=1e-6)

    #: some basic testing of the generated user.log file, because running the whole pipeline again
    #: will slow down the execution of the test suite.
    #: It will be worth moving the unit tests for the logger to a separate file
    #: once we generate different log files for each MPI process and we can compare them.
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    user_log_file = list(filter(lambda x: "user.log" in x, files))
    assert len(verbose_log_file) == 1
    assert len(user_log_file) == 1

    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert f"{user_log_file[0]}" in verbose_log_contents
    assert "The full dataset shape is (220, 128, 160)" in verbose_log_contents
    assert "Loading data: tests/test_data/tomo_standard.nxs" in verbose_log_contents
    assert "Path to data: entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 57:60, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 3, 160) of type uint16" in verbose_log_contents


def test_run_pipeline_cpu1_yaml(
    get_files: Callable, cmd, standard_data, yaml_cpu_pipeline1, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, yaml_cpu_pipeline1)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 132  # 128 images + yaml, 2 logfiles, intermdiate

    _check_tif(files, 128, (160, 160))

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
    assert "Path to data: entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 0:128, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 128, 160) of type uint16" in verbose_log_contents


def test_run_pipeline_cpu2_yaml(
    get_files: Callable, cmd, standard_data, yaml_cpu_pipeline2, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, yaml_cpu_pipeline2)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 34

    _check_tif(files, 30, (160, 160))

    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    user_log_file = list(filter(lambda x: "user.log" in x, files))
    assert len(verbose_log_file) == 1
    assert len(user_log_file) == 1

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    for file_to_open in h5_files:
        if "tomopy-recon-tomo-gridrec.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (160, 30, 160)
                assert f["data"].dtype == np.float32
                assert_allclose(np.sum(f["data"]), 694.70306, atol=1e-6, rtol=1e-6)

    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert f"{user_log_file[0]}" in verbose_log_contents
    assert "The full dataset shape is (220, 128, 160)" in verbose_log_contents
    assert "Loading data: tests/test_data/tomo_standard.nxs" in verbose_log_contents
    assert "Path to data: entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 30:60, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 30, 160) of type uint16" in verbose_log_contents


def test_run_pipeline_cpu3_yaml(
    get_files: Callable, cmd, standard_data, yaml_cpu_pipeline3, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, yaml_cpu_pipeline3)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 132  # 128 images + yaml, 2 logfiles, intermdiate

    _check_tif(files, 128, (160, 160))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    user_log_file = list(filter(lambda x: "user.log" in x, files))
    assert len(verbose_log_file) == 1
    assert len(user_log_file) == 1
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert f"{user_log_file[0]}" in verbose_log_contents
    assert "The full dataset shape is (220, 128, 160)" in verbose_log_contents
    assert "Loading data: tests/test_data/tomo_standard.nxs" in verbose_log_contents
    assert "Path to data: entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 0:128, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 128, 160) of type uint16" in verbose_log_contents
    assert " Global min -0.014979" in verbose_log_contents
    assert " Global max 0.04177" in verbose_log_contents
    assert " Global mean 0.0016174" in verbose_log_contents


def test_run_pipeline_cpu4_yaml(
    get_files: Callable, cmd, standard_data, yaml_cpu_pipeline4, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, yaml_cpu_pipeline4)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 132  # 128 images + yaml, 2 logfiles, intermdiate

    _check_tif(files, 128, (160, 160))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

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
    assert "Path to data: entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 0:128, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 128, 160) of type uint16" in verbose_log_contents


def test_run_pipeline_cpu5_yaml(
    get_files: Callable, cmd, standard_data, yaml_cpu_pipeline5, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, yaml_cpu_pipeline5)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 3

    # explore the debug log and the previewed data
    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert "Preview: (0:180, 32:96, 29:129)" in verbose_log_contents
    assert "Data shape is (180, 64, 100) of type uint16" in verbose_log_contents


@pytest.mark.cupy
def test_run_pipeline_gpu1_yaml(
    get_files: Callable, cmd, standard_data, yaml_gpu_pipeline1, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, yaml_gpu_pipeline1)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 132

    _check_tif(files, 128, (160, 160))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    for file_to_open in h5_files:
        if "httomolibgpu-FBP-tomo.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (160, 128, 160)
                assert f["data"].dtype == np.float32
                assert_allclose(np.sum(f["data"]), 2615.7332, atol=1e-6, rtol=1e-6)

    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    user_log_file = list(filter(lambda x: "user.log" in x, files))
    assert len(verbose_log_file) == 1
    assert len(user_log_file) == 1
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert f"{user_log_file[0]}" in verbose_log_contents
    assert "The full dataset shape is (220, 128, 160)" in verbose_log_contents
    assert "Loading data: tests/test_data/tomo_standard.nxs" in verbose_log_contents
    assert "Path to data: entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Preview: (0:180, 0:128, 0:160)" in verbose_log_contents
    assert "Data shape is (180, 128, 160) of type uint16" in verbose_log_contents
    assert "The amount of the available GPU memory is" in verbose_log_contents
    assert (
        "Using GPU 0 to transfer data of shape (180, 128, 160)" in verbose_log_contents
    )


def test_tomo_standard_testing_pipeline_output_with_save_all(
    get_files: Callable,
    cmd,
    standard_data,
    standard_loader,
    testing_pipeline,
    output_folder,
    merge_yamls,
):
    cmd.insert(7, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(8, "temp.yaml")
    cmd.insert(9, output_folder)
    subprocess.check_output(cmd)

    files = get_files("output_dir/")
    assert len(files) == 11

    _check_yaml(files, "temp.yaml")
    _check_tif(files, 3, (160, 160))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 5

    for file_to_open in h5_files:
        if "tomopy-recon-tomo-gridrec.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (160, 3, 160)
                assert f["data"].dtype == np.float32
                assert_allclose(np.mean(f["data"]), 0.0015362317, atol=1e-6, rtol=1e-6)
                assert_allclose(np.sum(f["data"]), 117.9826, atol=1e-6, rtol=1e-6)


def test_i12_testing_pipeline_output(
    get_files: Callable,
    cmd,
    i12_data,
    i12_loader,
    testing_pipeline,
    output_folder,
    merge_yamls,
):
    cmd.insert(7, i12_data)
    merge_yamls(i12_loader, testing_pipeline)
    cmd.insert(8, "temp.yaml")
    cmd.insert(9, output_folder)
    subprocess.check_output(cmd)

    files = get_files("output_dir/")
    assert len(files) == 18

    _check_yaml(files, "temp.yaml")

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2

    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 10

    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 5

    gridrec_recon = list(filter(lambda x: "recon-gridrec.h5" in x, h5_files))[0]
    minus_log_tomo = list(filter(lambda x: "minus_log.h5" in x, h5_files))[0]
    remove_stripe_fw_tomo = list(
        filter(lambda x: "remove_stripe_fw.h5" in x, h5_files)
    )[0]
    normalize_tomo = list(filter(lambda x: "normalize.h5" in x, h5_files))[0]

    with h5py.File(gridrec_recon, "r") as f:
        assert f["data"].shape == (192, 10, 192)
        assert_allclose(np.sum(f["data"]), 2157.03, atol=1e-2, rtol=1e-6)
        assert_allclose(np.mean(f["data"]), 0.0058513316, atol=1e-6, rtol=1e-6)
    with h5py.File(minus_log_tomo, "r") as f:
        assert_allclose(np.sum(f["data"]), 1756628.4, atol=1e-6, rtol=1e-6)
        assert_allclose(np.mean(f["data"]), 1.2636887, atol=1e-6, rtol=1e-6)
    with h5py.File(remove_stripe_fw_tomo, "r") as f:
        assert_allclose(np.sum(f["data"]), 1766357.8, atol=1e-6, rtol=1e-6)
        assert_allclose(np.mean(f["data"]), 1.2706878, atol=1e-6, rtol=1e-6)
    with h5py.File(normalize_tomo, "r") as f:
        assert f["data"].shape == (724, 10, 192)
        assert_allclose(np.sum(f["data"]), 393510.72, atol=1e-6, rtol=1e-6)
        assert_allclose(np.mean(f["data"]), 0.28308493, atol=1e-6, rtol=1e-6)

    concise_log_file = list(filter(lambda x: "user.log" in x, files))
    concise_log_contents = _get_log_contents(concise_log_file[0])
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert "The center of rotation is 95.5" in concise_log_contents
    assert "The full dataset shape is (724, 10, 192)" in verbose_log_contents
    assert (
        "Loading data: tests/test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
        in verbose_log_contents
    )
    assert "Path to data: /1-TempPlugin-tomo/data" in verbose_log_contents
    assert "Preview: (0:724, 0:10, 0:192)" in verbose_log_contents


# TODO: Add back in when ignoring darks/flats is added to the new loader
#
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
#
#     files = get_files("output_dir/")
#     assert len(files) == 16
#
#     _check_yaml(files, "temp.yaml")
#
#     log_files = list(filter(lambda x: ".log" in x, files))
#     assert len(log_files) == 1
#
#     tif_files = list(filter(lambda x: ".tif" in x, files))
#     assert len(tif_files) == 10
#
#     h5_files = list(filter(lambda x: ".h5" in x, files))
#     assert len(h5_files) == 4
#
#     log_contents = _get_log_contents(log_files[0])
#     assert "The full dataset shape is (724, 10, 192)" in log_contents
#     assert (
#         "Loading data: tests/test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
#         in log_contents
#     )
#     assert "Path to data: /1-TempPlugin-tomo/data" in log_contents
#     assert "Preview: (0:724, 0:10, 0:192)" in log_contents
#     assert "Running save_task_1 (pattern=projection): save_intermediate_data..." in log_contents
#     assert "Running save_task_2 (pattern=projection): save_intermediate_data..." in log_contents
#     assert "Running save_task_4 (pattern=sinogram): save_intermediate_data..." in log_contents
#     assert "The center of rotation for sinogram is 95.5" in log_contents
#     assert "Running save_task_5 (pattern=sinogram): save_intermediate_data..." in log_contents


def test_diad_testing_pipeline_output(
    get_files: Callable,
    cmd,
    diad_data,
    diad_loader,
    testing_pipeline,
    output_folder,
    merge_yamls,
):
    cmd.insert(7, diad_data)
    merge_yamls(diad_loader, testing_pipeline)
    cmd.insert(8, "temp.yaml")
    cmd.insert(9, output_folder)
    subprocess.check_output(cmd)

    files = get_files("output_dir/")
    assert len(files) == 10

    _check_yaml(files, "temp.yaml")
    _check_tif(files, 2, (26, 26))

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 5

    for file_to_open in h5_files:
        if "tomopy-normalize-tomo.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (3001, 2, 26)
                assert f["data"].dtype == np.float32
                assert_allclose(np.mean(f["data"]), 0.847944, atol=1e-6, rtol=1e-6)
                assert_allclose(np.sum(f["data"]), 132323.36, atol=1e-6, rtol=1e-6)
        if "tomopy-recon-tomo-gridrec.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (26, 2, 26)
                assert_allclose(np.mean(f["data"]), 0.005883, atol=1e-6, rtol=1e-6)
                assert_allclose(np.sum(f["data"]), 7.954298, atol=1e-6, rtol=1e-6)

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert "The full dataset shape is (3201, 22, 26)" in verbose_log_contents
    assert (
        "Loading data: tests/test_data/k11_diad/k11-18014.nxs" in verbose_log_contents
    )
    assert "Path to data: /entry/imaging/data" in verbose_log_contents
    assert "Preview: (100:3101, 5:7, 0:26)" in verbose_log_contents
    assert "Data shape is (3001, 2, 26) of type uint16" in verbose_log_contents


@pytest.mark.cupy
def test_run_diad_pipeline_gpu(
    get_files: Callable, cmd, diad_data, diad_pipeline_gpu, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, diad_data)
    cmd.insert(7, diad_pipeline_gpu)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 11

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert "The full dataset shape is (3201, 22, 26)" in verbose_log_contents
    assert (
        "Loading data: tests/test_data/k11_diad/k11-18014.nxs" in verbose_log_contents
    )
    assert "Path to data: /entry/imaging/data" in verbose_log_contents
    assert "Preview: (100:3101, 8:15, 0:26)" in verbose_log_contents
    assert "Data shape is (3001, 7, 26) of type uint16" in verbose_log_contents


@pytest.mark.cupy
def test_run_pipeline_360deg_gpu2(
    get_files: Callable, cmd, data360, yaml_gpu_pipeline360_2, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, data360)
    cmd.insert(7, yaml_gpu_pipeline360_2)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 7

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert "The full dataset shape is (3751, 3, 2560)" in verbose_log_contents
    assert "Loading data: tests/test_data/360scan/360scan.hdf" in verbose_log_contents
    assert "Path to data: entry1/tomo_entry/data/data" in verbose_log_contents
    assert "Data shape is (3601, 3, 2560) of type uint16" in verbose_log_contents


@pytest.mark.cupy
def test_run_gpu_pipeline_sweep_cor(
    get_files: Callable, cmd, standard_data, yaml_gpu_pipeline_sweep_cor, output_folder
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, yaml_gpu_pipeline_sweep_cor)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 9

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 0

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert "Data shape is (180, 7, 160) of type uint16" in verbose_log_contents
    assert "Total number of values across all processes: 6" in verbose_log_contents
    assert "Values executed in this process: 6" in verbose_log_contents


@pytest.mark.cupy
def test_run_gpu_pipeline_sweep_paganin(
    get_files: Callable,
    cmd,
    standard_data,
    yaml_gpu_pipeline_sweep_paganin,
    output_folder,
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    cmd.insert(7, yaml_gpu_pipeline_sweep_paganin)
    cmd.insert(8, output_folder)
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = get_files("output_dir/")
    assert len(files) == 104

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2
    verbose_log_file = list(filter(lambda x: "debug.log" in x, files))
    verbose_log_contents = _get_log_contents(verbose_log_file[0])

    assert "Data shape is (180, 7, 160) of type uint16" in verbose_log_contents
    assert "Total number of values across all processes: 50" in verbose_log_contents
    assert "Values executed in this process: 50" in verbose_log_contents
    assert "Parameter name: alpha" in verbose_log_contents
