import sys
from datetime import datetime
from os import mkdir
from pathlib import Path

from mpi4py import MPI
import numpy as np
from nvtx import annotate
import multiprocessing

from httomo.common import PipelineTasks

from httomo.data.hdf.loaders import standard_tomo
from httomo.data.hdf._utils.save import intermediate_dataset
from httomo.data.hdf._utils import reslice

from httomo.misc.corr import median_filter3d
from wrappers.tomopy.prep import normalize
from wrappers.tomopy.prep import stripe
from wrappers.tomopy.recon import rotation, algorithm


def cpu_pipeline(
    in_file: Path,
    out_dir: Path,
    data_key: str,
    dimension: int,
    crop: int = 100,
    pad: int = 0,
    ncores: int = 1,
    stop_after: PipelineTasks = PipelineTasks.RECONSTRUCT,
):
    """Run the CPU pipline to reconstruct the data.

    Args:
        in_file: The file to read data from.
        out_dir: The directory to write data to.
        data_key: The input file dataset key to read.
        dimension: The dimension to slice in.
        crop: The percentage of data to use. Defaults to 100.
        pad: The padding size to use. Defaults to 0.
        ncores: The number of the CPU cores per process
        stop_after: The stage after which the pipeline should stop. Defaults to
            PipelineStages.RECONSTRUCT.
    """
    comm = MPI.COMM_WORLD
    run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )
    if comm.rank == 0:
        mkdir(run_out_dir)
    if comm.size == 1:
        ncores = multiprocessing.cpu_count() # use all available CPU cores if not an MPI run

    ###################################################################################
    #                                 Loading the data
    with annotate(PipelineTasks.LOAD.name):
        (
            data,
            flats,
            darks,
            angles_radians,
            angles_total,
            detector_y,
            detector_x,
        ) = standard_tomo(in_file, data_key, dimension, crop, pad, comm)
    if stop_after == PipelineTasks.LOAD:
        sys.exit()      
    ###################################################################################
    #            3D median or dezinger filter to apply to raw data/flats/darks
    method_name = 'larix'
    save_result = True
    radius_kernel = 1 # a half-size of the median smoothing kernel
    mu_dezinger = 0.0 # when > 0.0, then dezinging enabled, otherwise median filter (smaller value more sensitive)
    with annotate(PipelineTasks.FILTER.name):
        data, flats, darks = larix(data, flats, darks, radius_kernel, mu_dezinger, ncores, comm)
    if save_result:
        intermediate_dataset(data, run_out_dir, method_name, comm)
    if stop_after == PipelineTasks.FILTER:
        sys.exit()
    ###################################################################################
    #                 Normalising the data and taking the negative log
    method_name = 'normalisation'
    save_result = True
    with annotate(PipelineTasks.NORMALIZE.name):
        data = normalize(data, flats, darks, ncores)
    if save_result:
        intermediate_dataset(data, run_out_dir, method_name, comm)
    if stop_after == PipelineTasks.NORMALIZE:
        sys.exit()
    ###################################################################################
    #                                 Removing stripes
    method_name = 'remove_stripe_fw' # TomoPy methods exposed here as in tomopy.prep.stripe 
    save_result = True
    with annotate(PipelineTasks.STRIPES.name):
        stripe(data, method_name, ncores)
    if save_result:
        intermediate_dataset(data, run_out_dir, method_name, comm)
    if stop_after == PipelineTasks.STRIPES:
        sys.exit()
    ###################################################################################
    #                        Calculating the center of rotation
    method_name = 'find_center_vo'
    with annotate(PipelineTasks.CENTER.name):
        rot_center = rotation(data, method_name, comm)
    if stop_after == PipelineTasks.CENTER:
        sys.exit()
    ###################################################################################
    #                    Saving/reloading the intermediate dataset
    with annotate(PipelineTasks.RESLICE.name):
        data, dimension = reslice(
            data, run_out_dir, dimension, angles_total, detector_y, detector_x, comm
        )
    if stop_after == PipelineTasks.RESLICE:
        sys.exit()
    ###################################################################################
    #                           Reconstruction with gridrec
    method_name = 'gridrec'
    with annotate(PipelineTasks.RECONSTRUCT.name):
        recon = algorithm(data, method_name, angles_radians, rot_center)
    if save_result:
        with annotate(PipelineTasks.SAVE.name):
            intermediate_dataset(recon, run_out_dir, method_name, comm)
    if stop_after == PipelineTasks.RECONSTRUCT:
        sys.exit()
