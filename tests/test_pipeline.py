import glob
import os
import re
import subprocess

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


def compare_two_yamls(original_yaml, copied_yaml):
    with open(original_yaml, "r") as oy, open(copied_yaml, "r") as cy:
        return oy.read() == cy.read()


def test_tomo_standard_testing_pipeline_output(
    cmd, standard_data, standard_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.pop(4)  #: don't save all
    cmd.insert(6, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(7, "temp.yaml")
    subprocess.check_output(cmd)

    # recurse through output_dir and check that all files are there
    files = read_folder("output_dir/")
    assert len(files) == 6

    # check that the contents of the copied YAML in the output directory matches
    # the contents of the input YAML
    copied_yaml_path = list(filter(lambda x: ".yaml" in x, files)).pop()
    assert compare_two_yamls("temp.yaml", copied_yaml_path)

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


def test_tomo_standard_testing_pipeline_output_with_save_all(
    cmd, standard_data, standard_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.insert(7, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(8, "temp.yaml")
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 9

    # check that the contents of the copied YAML in the output directory matches
    # the contents of the input YAML
    copied_yaml_path = list(filter(lambda x: ".yaml" in x, files)).pop()
    assert compare_two_yamls("temp.yaml", copied_yaml_path)

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
def test_gpu_pipeline_output_with_save_all(
    cmd, standard_data, gpu_pipeline, output_folder
):
    cmd.insert(7, standard_data)
    cmd.insert(8, gpu_pipeline)
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 15

    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 10
    total_sum = 0
    for i in range(10):
        arr = np.array(Image.open(tif_files[i]))
        assert arr.dtype == np.uint8
        assert arr.shape == (160, 160)
        total_sum += arr.sum()

    assert total_sum == 17871073.0

    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 3
    with h5py.File(h5_files[0], "r") as f:
        assert f["data"].shape == (180, 10, 160)
        assert_allclose(np.sum(f["data"]), 84874.38, atol=1e-5)
        assert_allclose(np.mean(f["data"]), 0.2947027, atol=1e-5)
    with h5py.File(h5_files[2], "r") as f:
        assert_allclose(np.sum(f["data"]), 84127.195, atol=1e-5)
        assert_allclose(np.mean(f["data"]), 0.29210833, atol=1e-5)
    with h5py.File(h5_files[1], "r") as f:
        assert_allclose(np.sum(f["data"]), 210.13103, atol=1e-5)
        assert_allclose(np.mean(f["data"]), 0.00082082435, atol=1e-5)
        assert f["data"].shape == (10, 160, 160)


def test_i12_testing_pipeline_output(
    cmd, i12_data, i12_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.insert(7, i12_data)
    merge_yamls(i12_loader, testing_pipeline)
    cmd.insert(8, "temp.yaml")
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 16

    copied_yaml_path = list(filter(lambda x: ".yaml" in x, files)).pop()
    assert compare_two_yamls("temp.yaml", copied_yaml_path)

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 1

    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 10
    total_sum = 0
    for i in range(10):
        arr = np.array(Image.open(tif_files[i]))
        assert arr.dtype == np.uint8
        assert arr.shape == (192, 192)
        total_sum += arr.sum()

    assert total_sum == 25834156.0

    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 4
    with h5py.File(h5_files[0], "r") as f:
        assert f["data"].shape == (724, 10, 192)
        assert_allclose(np.sum(f["data"]), 393510.72, atol=1e-6)
        assert_allclose(np.mean(f["data"]), 0.28308493, atol=1e-6)
    with h5py.File(h5_files[1], "r") as f:
        assert_allclose(np.sum(f["data"]), 1756628.4, atol=1e-6)
        assert_allclose(np.mean(f["data"]), 1.2636887, atol=1e-6)
    with h5py.File(h5_files[2], "r") as f:
        assert_allclose(np.sum(f["data"]), 1766357.8, atol=1e-6)
        assert_allclose(np.mean(f["data"]), 1.2706878, atol=1e-6)
    with h5py.File(h5_files[3], "r") as f:
        assert f["data"].shape == (10, 192, 192)
        assert_allclose(np.sum(f["data"]), 2157.035, atol=1e-6)
        assert_allclose(np.mean(f["data"]), 0.0058513316, atol=1e-6)

    log_contents = _get_log_contents(log_files[0])
    assert "DEBUG | The full dataset shape is (724, 10, 192)" in log_contents
    assert (
        "DEBUG | Loading data: tests/test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
        in log_contents
    )
    assert "DEBUG | Path to data: /1-TempPlugin-tomo/data" in log_contents
    assert "DEBUG | Preview: (0:724, :, :)" in log_contents
    assert "Saving intermediate file: 2-tomopy-normalize-tomo.h5" in log_contents
    assert "Saving intermediate file: 3-tomopy-minus_log-tomo.h5" in log_contents
    assert "Reslicing not necessary, as there is only one process" in log_contents
    assert "Saving intermediate file: 4-tomopy-remove_stripe_fw-tomo.h5" in log_contents
    assert "The center of rotation for 180 degrees sinogram is 95.5" in log_contents
    assert "Saving intermediate file: 6-tomopy-recon-tomo-gridrec.h5" in log_contents
    assert "INFO | ~~~ Pipeline finished ~~~" in log_contents


def test_diad_testing_pipeline_output(
    cmd, diad_data, diad_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.insert(7, diad_data)
    merge_yamls(diad_loader, testing_pipeline)
    cmd.insert(8, "temp.yaml")
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 8

    # check that the contents of the copied YAML in the output directory matches
    # the contents of the input YAML
    copied_yaml_path = list(filter(lambda x: ".yaml" in x, files)).pop()
    assert compare_two_yamls("temp.yaml", copied_yaml_path)

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


def test_sweep_pipeline_with_save_all_using_mpi(
    cmd, standard_data, sample_pipelines, standard_loader, output_folder
):
    #: - - - - - - - - - - SERIAL RUN - - - - - - - - - - - - - - - - -
    pipeline = sample_pipelines + "testing/sweep_testing_pipeline.yaml"
    cmd.insert(7, standard_data)
    cmd.insert(8, pipeline)
    subprocess.check_output(cmd)

    #: - - - - - - - - - -  PARALLEL RUN - - - - - - - - - - -
    local.cmd.mpirun("-n", "4", *cmd)

    #: - - - - - - - - - - SERIAL vs PARALLEL OUTPUT - - - - - - -
    files = read_folder("output_dir/")
    assert len(files) == 12

    copied_yaml_path = list(filter(lambda x: ".yaml" in x, files))
    assert compare_two_yamls(pipeline, copied_yaml_path[0])
    assert compare_two_yamls(pipeline, copied_yaml_path[1])

    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 4

    #: check that the image size is correct
    imarray = np.array(Image.open(tif_files[0]))
    mpi_imarray = np.array(Image.open(tif_files[2]))
    assert imarray.shape == (128, 160) == mpi_imarray.shape
    assert imarray.sum() == 3856477 == mpi_imarray.sum()

    imarray = np.array(Image.open(tif_files[1]))
    mpi_imarray = np.array(Image.open(tif_files[3]))
    assert imarray.shape == (128, 160) == mpi_imarray.shape
    assert imarray.sum() == 3855857 == mpi_imarray.sum()

    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 4
    with h5py.File(h5_files[0], "r") as f, h5py.File(h5_files[2], "r") as f2:
        assert (
            f["/data/param_sweep_0"].shape
            == (180, 128, 160)
            == f2["/data/param_sweep_0"].shape
        )
        assert (
            f["/data/param_sweep_0"].dtype
            == np.float32
            == f2["/data/param_sweep_0"].dtype
        )

        s = np.sum(f["/data/param_sweep_0"])
        assert_allclose(s, 2981532700.0, atol=1e-6)
        assert_allclose(np.sum(f2["/data/param_sweep_0"]), s, atol=1e-6)

        m = np.mean(f["/data/param_sweep_0"])
        assert_allclose(m, 808.7925, atol=1e-6)
        assert_allclose(np.mean(f2["/data/param_sweep_0"]), m, atol=1e-6)

    with h5py.File(h5_files[1], "r") as f, h5py.File(h5_files[3], "r") as f2:
        assert (
            f["/data/param_sweep_1"].shape
            == (180, 128, 160)
            == f2["/data/param_sweep_1"].shape
        )
        assert (
            f["/data/param_sweep_1"].dtype
            == np.float32
            == f2["/data/param_sweep_1"].dtype
        )

        s = np.sum(f["/data/param_sweep_1"])
        assert_allclose(s, 3053065.5, atol=1e-6)
        assert_allclose(np.sum(f2["/data/param_sweep_1"]), s, atol=1e-6)

        m = np.mean(f["/data/param_sweep_1"])
        assert_allclose(m, 0.828197, atol=1e-6)
        assert_allclose(np.mean(f2["/data/param_sweep_1"]), m, atol=1e-6)

    log_files = list(filter(lambda x: ".log" in x, files))
    assert len(log_files) == 2

    log_contents = _get_log_contents(log_files[0])
    mpi_log_contents = _get_log_contents(log_files[1])
    assert "DEBUG | The full dataset shape is (220, 128, 160)" in log_contents
    assert (
        "DEBUG | RANK: [0], Data shape is (180, 128, 160) of type uint16"
        in log_contents
    )
    assert "DEBUG | Preview: (0:180, :, :)" in log_contents
    assert "INFO | ~~~ Pipeline finished ~~~" in log_contents

    #: user log and mpi log would differ in the data shapes
    assert (
        "DEBUG | RANK: [0], Data shape is (45, 128, 160) of type uint16"
        in mpi_log_contents
    )


def test_sweep_range_pipeline_with_step_absent(
    cmd, standard_data, sample_pipelines, output_folder
):
    cmd.insert(7, standard_data)
    cmd.insert(8, sample_pipelines + "testing/step_absent.yml")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    log_files = list(filter(lambda x: ".log" in x, read_folder("output_dir/")))
    assert len(log_files) == 1

    log_contents = _get_log_contents(log_files[0])

    assert (
        "ERROR | Please provide `start`, `stop`, `step` values"
        " when specifying a range to peform a parameter sweep over."
    ) in log_contents


@pytest.mark.cupy
def test_multi_inputs_pipeline(cmd, standard_data, sample_pipelines, output_folder):
    cmd.insert(7, standard_data)
    cmd.insert(8, sample_pipelines + "multi_inputs/01_dezing_multi_inputs.yaml")
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 5

    copied_yaml_path = list(filter(lambda x: ".yaml" in x, files)).pop()
    assert compare_two_yamls(
        sample_pipelines + "multi_inputs/01_dezing_multi_inputs.yaml", copied_yaml_path
    )

    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 3

    with h5py.File(h5_files[0], "r") as f:
        assert f["data"].shape == (180, 128, 160)
        assert f["data"].dtype == np.uint16
        assert np.sum(f["data"]) == 2981388880
        assert_allclose(np.mean(f["data"]), 808.7534939236111, atol=1e-6)
    with h5py.File(h5_files[1], "r") as f:
        assert f["data"].shape == (20, 128, 160)
        assert f["data"].dtype == np.uint16
        assert np.sum(f["data"]) == 399933384
        assert_allclose(np.mean(f["data"]), 976.39986328125, atol=1e-6)
    with h5py.File(h5_files[2], "r") as f:
        assert np.sum(f["data"]) == 0
