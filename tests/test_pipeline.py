import os
import subprocess

import h5py
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
import glob


def read_folder(folder):
    files = []
    for f in os.listdir(folder):
        f = os.path.join(folder, f)
        if os.path.isdir(f):
            files = [*files, *read_folder(f)]
        else:
            files.append(f)
    return files


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
    assert len(files) == 4

    # check the .tif files
    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 3
    #: check that the image size is correct
    imarray = np.array(Image.open(tif_files[0]))
    assert imarray.shape == (160, 160)

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 1

    for name in h5_files:
        if "tomopy-recon-tomo-gridrec.h5" in name:
            file_to_open = name
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (3, 160, 160)
                assert f["data"].dtype == np.float32
                assert_allclose(np.mean(f["data"]), -8.037842e-06, atol=1e-6)
                assert_allclose(np.sum(f["data"]), -0.617306, atol=1e-6)


def test_tomo_standard_testing_pipeline_output_with_save_all(
    cmd, standard_data, standard_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.insert(6, standard_data)
    merge_yamls(standard_loader, testing_pipeline)
    cmd.insert(7, "temp.yaml")
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 7

    # check the .tif files
    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 3

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 4

    for name in h5_files:
        if "tomopy-normalize-tomo.h5" in name:
            file_to_open = name
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (180, 3, 160)
                assert f["data"].dtype == np.float32
                assert_allclose(np.mean(f["data"]), 1.004919, atol=1e-6)
                assert_allclose(np.sum(f["data"]), 86824.984, atol=1e-6)
        if "tomopy-minus_log-tomo.h5" in name:
            file_to_open = name
            with h5py.File(file_to_open, "r") as f:
                assert f["data"].shape == (180, 3, 160)
                assert_allclose(np.mean(f["data"]), -0.004374, atol=1e-6)
                assert_allclose(np.sum(f["data"]), -377.88608, atol=1e-6)
        if "tomopy-remove_stripe_fw-tomo.h5" in name:
            file_to_open = name
            with h5py.File(file_to_open, "r") as f:
                assert_allclose(np.mean(f["data"]), -0.004198, atol=1e-6)
                np.testing.assert_almost_equal(np.sum(f["data"]), -362.73358, decimal=4)


def test_diad_testing_pipeline_output(
    cmd, diad_data, diad_loader, testing_pipeline, output_folder, merge_yamls
):
    cmd.insert(6, diad_data)
    merge_yamls(diad_loader, testing_pipeline)
    cmd.insert(7, "temp.yaml")
    subprocess.check_output(cmd)

    files = read_folder("output_dir/")
    assert len(files) == 6

    #: check the .tif files
    tif_files = list(filter(lambda x: ".tif" in x, files))
    assert len(tif_files) == 2

    #: check that the image size is correct
    imarray = np.array(Image.open(tif_files[0]))
    assert imarray.shape == (26, 26)

    #: check the generated h5 files
    h5_files = list(filter(lambda x: ".h5" in x, files))
    assert len(h5_files) == 4

    for name in h5_files:
        if "tomopy-normalize-tomo.h5" in name:
            file_to_open = name
    with h5py.File(file_to_open, "r") as f:
        assert f["data"].shape == (3001, 2, 26)
        assert f["data"].dtype == np.float32
        assert_allclose(np.mean(f["data"]), 0.847944, atol=1e-6)
        assert_allclose(np.sum(f["data"]), 132323.36, atol=1e-6)

    for name in h5_files:
        if "tomopy-recon-tomo-gridrec.h5" in name:
            file_to_open = name
    with h5py.File(file_to_open, "r") as f:
        assert f["data"].shape == (2, 26, 26)
        assert_allclose(np.mean(f["data"]), 0.005883, atol=1e-6)
        assert_allclose(np.sum(f["data"]), 7.954298, atol=1e-6)
