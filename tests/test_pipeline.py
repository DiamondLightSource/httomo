import os
import re
import subprocess

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import Image
import glob

PATTERN = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")


def _get_log_contents(file):
    """return contents of the user.log file"""
    with open(file, "r") as f:
        log_contents = f.read()

    #: check that the generated log file has no ansi escape sequence
    assert not PATTERN.search(log_contents)

    return log_contents


def read_folder(folder):
    files = []
    for f in os.listdir(folder):
        f = os.path.join(folder, f)
        if os.path.isdir(f):
            files = [*files, *read_folder(f)]
        else:
            files.append(f)
    return files


@pytest.mark.cupy
def test_tomo_standard_testing_pipeline_output(
    cmd, standard_data, standard_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.pop(3)  #: don't save all
    cmd.insert(5, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(6, "temp.yaml")
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = read_folder("output_dir/")
    assert len(files) == 5

    # check the .tif files
    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 3
    #: check that the image size is correct
    imarray = np.array(Image.open(tif_files[0]))
    assert imarray.shape == (160, 160)

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    for file_to_open in h5_files:
        if "tomopy-recon-tomo-gridrec.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (3, 160, 160)
                assert f["data"].dtype == np.float32
                assert_allclose(np.mean(f["data"]), -8.037842e-06, atol=1e-6)
                assert_allclose(np.sum(f["data"]), -0.617306, atol=1e-6)

    #: some basic testing of the generated user.log file, because running the whole pipeline again
    #: will slow down the execution of the test suite.
    #: It will be worth moving the unit tests for the logger to a separate file
    #: once we generate different log files for each MPI process and we can compare them.
    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 1

    log_contents = _get_log_contents(log_files[0])

    assert f"INFO | See the full log file at: {log_files[0]}" in log_contents
    assert "DEBUG | The full dataset shape is (220, 128, 160)" in log_contents
    assert "DEBUG | Loading data: tests/test_data/tomo_standard.nxs" in log_contents
    assert "DEBUG | Path to data: entry1/tomo_entry/data/data" in log_contents
    assert "DEBUG | Preview: (0:180, 0:3:, :)" in log_contents
    assert (
        "DEBUG | RANK: [0], Data shape is (180, 3, 160) of type uint16" in log_contents
    )
    assert "DEBUG | <-------Reslicing/rechunking the data-------->" in log_contents
    assert "DEBUG | Total number of reslices: 1" in log_contents
    assert "INFO | ~~~ Pipeline finished ~~~" in log_contents


@pytest.mark.cupy
def test_tomo_standard_testing_pipeline_output_with_save_all(
    cmd, standard_data, standard_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.insert(6, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(7, "temp.yaml")
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 8

    # check the .tif files
    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 3

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 4

    for file_to_open in h5_files:
        if "tomopy-normalize-tomo.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (180, 3, 160)
                assert f["data"].dtype == np.float32
                assert_allclose(np.mean(f["data"]), 1.004919, atol=1e-6)
                assert_allclose(np.sum(f["data"]), 86824.984, atol=1e-6)
        if "tomopy-minus_log-tomo.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (180, 3, 160)
                assert_allclose(np.mean(f["data"]), -0.004374, atol=1e-6)
                assert_allclose(np.sum(f["data"]), -377.88608, atol=1e-6)
        if "tomopy-remove_stripe_fw-tomo.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert_allclose(np.mean(f["data"]), -0.004198, atol=1e-6)
                np.testing.assert_almost_equal(np.sum(f["data"]), -362.73358, decimal=4)


@pytest.mark.cupy
def test_diad_testing_pipeline_output(
    cmd, diad_data, diad_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.insert(6, diad_data)
    merge_yamls(diad_loader, testing_pipeline)
    cmd.insert(7, "temp.yaml")
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 7

    #: check the .tif files
    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 2

    #: check that the image size is correct
    imarray = np.array(Image.open(tif_files[0]))
    assert imarray.shape == (26, 26)

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 4

    for file_to_open in h5_files:
        if "tomopy-normalize-tomo.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (3001, 2, 26)
                assert f["data"].dtype == np.float32
                assert_allclose(np.mean(f["data"]), 0.847944, atol=1e-6)
                assert_allclose(np.sum(f["data"]), 132323.36, atol=1e-6)
        if "tomopy-recon-tomo-gridrec.h5" in file_to_open:
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (2, 26, 26)
                assert_allclose(np.mean(f["data"]), 0.005883, atol=1e-6)
                assert_allclose(np.sum(f["data"]), 7.954298, atol=1e-6)

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 1

    log_contents = _get_log_contents(log_files[0])

    assert f"INFO | See the full log file at: {log_files[0]}" in log_contents
    assert "DEBUG | The full dataset shape is (3201, 22, 26)" in log_contents
    assert (
        "DEBUG | Loading data: tests/test_data/k11_diad/k11-18014.nxs" in log_contents
    )
    assert "DEBUG | Path to data: /entry/imaging/data" in log_contents
    assert "DEBUG | Preview: (100:3101, 5:7:, :)" in log_contents
    assert (
        "DEBUG | RANK: [0], Data shape is (3001, 2, 26) of type uint16" in log_contents
    )
    assert (
        "DEBUG | Saving intermediate file: 2-tomopy-normalize-tomo.h5" in log_contents
    )
    assert (
        "DEBUG | Reslicing not necessary, as there is only one process" in log_contents
    )
    assert "INFO | ~~~ Pipeline finished ~~~" in log_contents


@pytest.mark.cupy
def test_sweep_range_pipeline_with_step_absent(
    cmd, standard_data, sample_pipelines, output_folder
):
    cmd.insert(6, standard_data)
    cmd.insert(7, sample_pipelines + "testing/step_absent.yml")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    log_files = list(filter(lambda x: ".log" in x, read_folder("output_dir/")))
    assert len(log_files) == 1

    log_contents = _get_log_contents(log_files[0])

    assert (
        "ERROR | Please provide `start`, `stop`, `step` values"
        " when specifying a range to peform a parameter sweep over."
    ) in log_contents
