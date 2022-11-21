import multiprocessing
import numpy as np
import os
import yaml

from datetime import datetime
from mpi4py import MPI
from os import mkdir

# absolute httomo/tomopy imports
from httomo.data.hdf.loaders import standard_tomo
from httomo.misc.segm import binary_thresholding


def test_binary_thresholding():
    #---- testing `binary_thresholding` on np.zeros ----#

    _data = np.zeros(shape=(100, 100, 100))
    _binary_mask = binary_thresholding(_data, 0.5)
    for _ in range(10):
        assert np.all(_binary_mask == 0)

    #---- testing `binary_thresholding` on tomo_standard ----#

    # load the corresponding YAML configuration task file which can be excecuted through the HTTomo task runner
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
    data, _, _, _, _, _, _ = standard_tomo(standard_tomo_params['name'],
                                           in_file, 
                                           standard_tomo_params['data_path'],
                                           standard_tomo_params['image_key_path'],
                                           standard_tomo_params['dimension'],
                                           standard_tomo_params['preview'],
                                           standard_tomo_params['pad'],
                                           comm)
    
    binary_mask = binary_thresholding(data, 0.1)
    for _ in range(10):
        assert np.all(binary_mask == 1)
