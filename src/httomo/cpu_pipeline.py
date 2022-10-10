import sys
from datetime import datetime
from os import mkdir
from pathlib import Path

from mpi4py import MPI
from nvtx import annotate

from httomo.common import PipelineTasks
from httomo.tasks.filtering.median3d import filter_data_larix
from httomo.tasks.centering.tomopy_cpu import find_center_of_rotation
from httomo.tasks.data_loading.original import load_data
from httomo.tasks.normalization.original_cpu import normalize_data
from httomo.tasks.reconstruction.tomopy_cpu import reconstruct
from httomo.tasks.reslice.original import reslice
from httomo.tasks.saving.original import save_data
from httomo.tasks.stripe_removal.original_cpu import remove_stripes


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
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_recon"
    )
    if comm.rank == 0:
        mkdir(run_out_dir)

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
        ) = load_data(in_file, data_key, dimension, crop, pad, comm)
    if stop_after == PipelineTasks.LOAD:
        sys.exit()
    ###################################################################################
    #                3D median or dezinger filter to apply to raw data/flats/darks
    with annotate(PipelineTasks.FILTER.name):
        radius_kernel = 1 
        mu_dezinger = 0.0 # when > 0.0, then dezinging enabled, otherwise median filter
        data, flats, darks = filter_data_larix(data, flats, darks, comm, radius_kernel, mu_dezinger, ncores)
    if stop_after == PipelineTasks.FILTER:
        sys.exit()
    ###################################################################################
    #                 Normalising the data and taking the negative log
    with annotate(PipelineTasks.NORMALIZE.name):
        data = normalize_data(data, flats, darks)
    if stop_after == PipelineTasks.NORMALIZE:
        sys.exit()
    ###################################################################################
    #                                 Removing stripes
    with annotate(PipelineTasks.STRIPES.name):
        remove_stripes(data)
    if stop_after == PipelineTasks.STRIPES:
        sys.exit()
    ###################################################################################
    #                        Calculating the center of rotation
    with annotate(PipelineTasks.CENTER.name):
        rot_center = find_center_of_rotation(data, comm)
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
    with annotate(PipelineTasks.RECONSTRUCT.name):
        recon = reconstruct(data, angles_radians, rot_center)
    if stop_after == PipelineTasks.RECONSTRUCT:
        sys.exit()
    ###################################################################################
    #                     Saving the result of the reconstruction
    with annotate(PipelineTasks.SAVE.name):
        save_data(recon, run_out_dir, comm)
