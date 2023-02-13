import os
import subprocess
import sys
from pathlib import Path
from shutil import rmtree

import numpy as np
from PIL import Image

if not os.path.exists("output_dir"):
    os.mkdir("output_dir/")


def read_folder(folder):
    files = []
    for f in os.listdir(folder):
        f = os.path.join(folder, f)
        if os.path.isdir(f):
            files = [*files, *read_folder(f)]
        else:
            files.append(f)
    return files


def test_tomo_standard_loaded(clean_folder):
    cmd = [
        sys.executable,
        "-m", "httomo",
        "testdata/tomo_standard.nxs",
        "samples/pipeline_template_examples/testing_pipeline.yaml",
        "output_dir/",
        "task_runner",
    ]
    output = subprocess.check_output(cmd).decode().strip()
    assert "Running task 1 (pattern=projection): standard_tomo" in output
    assert "Running task 2 (pattern=projection): normalize" in output
    assert "Running task 3 (pattern=projection): minus_log" in output
    assert "Running task 5 (pattern=sinogram): find_center_vo" in output
    assert "Running task 6 (pattern=sinogram): recon" in output
    assert "Running task 7 (pattern=all): save_to_images" in output
    assert "Total number of reslices: 1" in output

    # recurse through output_dir and check that all files are there
    files = read_folder("output_dir/")
    assert len(files) == 5

    # check the .tif files
    tif_files = list(filter(lambda x: '.tif' in x, files))
    assert len(tif_files) == 3
    #: check that the image size is correct
    imarray = np.array(Image.open(tif_files[0]))
    assert imarray.shape == (160, 160)

    #: check h5 files
    h5_files = list(filter(lambda x: '.h5' in x, files))
    assert len(h5_files) == 2


def test_tomo_standard_loaded_with_save_all(clean_folder):
    cmd = [
        sys.executable,
        "-m", "httomo",
        "--save_all",
        "testdata/tomo_standard.nxs",
        "samples/pipeline_template_examples/testing_pipeline.yaml",
        "output_dir/",
        "task_runner",
    ]
    output = subprocess.check_output(cmd).decode().strip()
    assert "Saving intermediate file: 2-tomopy-normalize-tomo.h5" in output
    assert "Saving intermediate file: 3-tomopy-minus_log-tomo.h5" in output
    assert "Saving intermediate file: 4-tomopy-remove_stripe_fw-tomo.h5" in output
    assert "Saving intermediate file: 6-tomopy-recon-tomo-gridrec.h5" in output

    files = read_folder("output_dir/")
    assert len(files) == 8

    # check the .tif files
    tif_files = list(filter(lambda x: '.tif' in x, files))
    assert len(tif_files) == 3

    #: check h5 files
    h5_files = list(filter(lambda x: '.h5' in x, files))
    assert len(h5_files) == 5


def test_k11_diad_loaded(clean_folder):
    cmd = [
        sys.executable,
        "-m", "httomo",
        "--save_all", "--ncore", "2",
        "testdata/k11_diad/k11-18014.nxs",
        "samples/beamline_loader_configs/diad.yaml",
        "output_dir/",
        "task_runner",
    ]
    output = subprocess.check_output(cmd).decode().strip()
    assert "dataset shape is (3201, 22, 26)" in output
    assert "testdata/k11_diad/k11-18014.nxs" in output
    assert "Data shape is (3001, 22, 26) of type uint16" in output
