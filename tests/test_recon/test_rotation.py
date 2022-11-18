import multiprocessing
import numpy as np
import os
import pprint
import sys
import tomopy
import yaml

from datetime import datetime
from mpi4py import MPI
from numpy.testing import assert_allclose
from os import mkdir

# absolute httomo/tomopy imports
from httomo.data.hdf.loaders import standard_tomo
from httomo.recon.rotation import find_center_360
from tomopy.prep.normalize import normalize, minus_log


def test_find_center_360():
    mat = np.ones(shape=(100, 100, 100))
    (cor, overlap, side, overlap_position) = find_center_360(mat[:, 2, :])
    for _ in range(10):
        assert_allclose(cor, 5.0)
        assert_allclose(overlap, 12.0)
        assert side == 0
        assert_allclose(overlap_position, 7.0)

    # load the corresponding YAML configuration ftask ile which can be excecuted through the HTTomo task runner
    pipeline = yaml.safe_load(open('../../pipeline_template_examples/02_basic_cpu_pipeline_tomo_standard.yaml'))
    pp = pprint.PrettyPrinter(indent=1)

    # set paths to tomo_standard
    current_dir = os.getcwd()
    in_file = os.path.join(current_dir, "../../../testdata" ,"tomo_standard.nxs")
    run_out_dir_main = os.path.join(current_dir, "../../../testdata" ,"output_temp")
    run_out_dir = os.path.join(current_dir, "../../../testdata" ,"output_temp",
                    f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output")

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        if not os.path.exists(run_out_dir_main):
            os.makedirs(run_out_dir_main)
        mkdir(run_out_dir)
    if comm.size == 1:
        ncore = multiprocessing.cpu_count()

    # getting the dictionaries from the YAML template file
    standard_tomo_params = pipeline[0]['httomo.data.hdf.loaders']['standard_tomo']

    # loading the data
    data, flats, darks, _, _, _, _ = standard_tomo(standard_tomo_params['name'],
                                                   in_file,
                                                   standard_tomo_params['data_path'],
                                                   standard_tomo_params['image_key_path'],
                                                   standard_tomo_params['dimension'],
                                                   standard_tomo_params['preview'],
                                                   standard_tomo_params['pad'],
                                                   comm)

    # Now, normalising raw data using TomoPy functions
    data = normalize(data, flats, darks, ncore=ncore, cutoff=10)
    data[data == 0.0] = 1e-09
    data = minus_log(data, ncore=ncore)

    eps = 1e-5
    (cor, overlap, side, overlap_position) = find_center_360(data[:, 10, :])
    assert_allclose(cor, 40.361228942871094, rtol=eps)
    assert_allclose(overlap, 82.72245788574219, rtol=eps)
    assert side == 0
    assert_allclose(overlap_position, 77.72245788574219, rtol=eps)
