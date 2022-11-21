import multiprocessing
import numpy as np
import os
import yaml

from datetime import datetime
from mpi4py import MPI
from numpy.testing import assert_allclose
from os import mkdir

# absolute httomo/tomopy imports
from httomo.data.hdf.loaders import standard_tomo
from httomo.prep.stripe import detect_stripes
from tomopy.prep.normalize import normalize, minus_log


def test_stripes():
    #---- testing `detect_stripes` on tomo_standard ---#

    # load the corresponding YAML configuration ftask ile which can be excecuted through the HTTomo task runner
    pipeline = yaml.safe_load(open('../../pipeline_template_examples/02_basic_cpu_pipeline_tomo_standard.yaml'))

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

    stripes_weight = detect_stripes(data, ncore=ncore)

    assert_allclose(np.mean(stripes_weight), 0.06417494)
    assert_allclose(np.sum(stripes_weight), 236574.5)
