import sys
from datetime import datetime
from os import mkdir
from pathlib import Path

import cupy
from mpi4py import MPI
from nvtx import annotate


from httomo.common import PipelineTasks

from httomo.tasks._DATA_.data_loading import load_data
from httomo.tasks._DATA_.data_saving import save_data
from httomo.tasks._DATA_.data_reslice import reslice
from httomo.tasks._DATA_.mpi_gatherv import to_device

from httomo.tasks._METHODS_.filtering.original_gpu import filter_data
from httomo.tasks._METHODS_.centering.original_gpu import find_center_of_rotation
from httomo.tasks._METHODS_.normalisation.original_gpu import normalise_data
from httomo.tasks._METHODS_.reconstruction.tomopy_gpu import reconstruct
from httomo.tasks._METHODS_.stripe_removal.original_gpu import remove_stripes


def gpu_pipeline(
    in_file: Path,
    out_dir: Path,
    data_key: str,
    dimension: int,
    crop: int = 100,
    pad: int = 0,
    stop_after: PipelineTasks = PipelineTasks.RECONSTRUCT,
):
    """Run the GPU pipline to reconstruct the data.

    Args:
        in_file: The file to read data from.
        out_dir: The directory to write data to.
        data_key: The input file dataset key to read.
        dimension: The dimension to slice in.
        crop: The percentage of data to use. Defaults to 100.
        pad: The padding size to use. Defaults to 0.
        stop_after: The stage after which the pipeline should stop. Defaults to
            PipelineStages.RECONSTRUCT.
    """
    comm = MPI.COMM_WORLD
    run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_recon"
    )
    if comm.rank == 0:
        mkdir(run_out_dir)
    num_GPUs = cupy.cuda.runtime.getDeviceCount()
    gpu_id = int(comm.rank / comm.size * num_GPUs)
    gpu_comm = comm.Split(gpu_id)
    proc_id = f"[{gpu_id}:{gpu_comm.rank}]"
    cupy.cuda.Device(gpu_id).use()

    ###################################################################################
    #                                 Loading the data
    with annotate(PipelineTasks.LOAD.name, color="blue"):
        (
            data,
            host_flats,
            host_darks,
            host_angles,
            angles_total,
            detector_y,
            detector_x,
        ) = load_data(in_file, data_key, dimension, crop, pad, comm)
    with annotate("HOST TO DEVICE", color="green"):
        flats = cupy.asarray(host_flats)
        darks = cupy.asarray(host_darks)
        angles = cupy.asarray(host_angles)
    if stop_after == PipelineTasks.LOAD:
        sys.exit()
    ###############################################################################
    #              3D Median filter to apply to raw data/flats/darks
    with annotate(PipelineTasks.FILTER.name, color="blue"):
        data, flats, darks = filter_data(data, flats, darks)
    print(f"{proc_id} Finished {PipelineTasks.FILTER.name}")
    if stop_after == PipelineTasks.FILTER:
        sys.exit()
    ###############################################################################
    #               Normalising the data and taking the negative log
    with annotate(PipelineTasks.NORMALIZE.name, color="blue"):
        data = normalise_data(data, darks, flats)
    print(f"{proc_id} Finished {PipelineTasks.NORMALIZE.name}")
    if stop_after == PipelineTasks.NORMALIZE:
        sys.exit()
    ###############################################################################
    #                               Removing stripes
    with annotate(PipelineTasks.STRIPES.name, color="blue"):
        data = remove_stripes(data)
    print(f"{proc_id} Finished {PipelineTasks.STRIPES.name}")
    if stop_after == PipelineTasks.STRIPES:
        sys.exit()
    ###############################################################################
    #                      Calculating the center of rotation
    with annotate(PipelineTasks.CENTER.name, color="blue"):
        rot_center = find_center_of_rotation(data)
    print(f"{proc_id} Finished {PipelineTasks.CENTER.name}")
    if stop_after == PipelineTasks.CENTER:
        sys.exit()
    ###############################################################################
    #                  Saving/reloading the intermediate dataset
    with annotate(PipelineTasks.RESLICE.name, color="blue"):
        with annotate("FROM DEVICE", color="green"):
            data = cupy.asnumpy(data)
        with annotate("I/O", color="green"):
            data, dimension = reslice(
                data, run_out_dir, dimension, len(angles), detector_y,
                detector_x, comm
            )
        with annotate("TO DEVICE", color="green"):
            data = cupy.asarray(data)
    print(f"{proc_id} Finished {PipelineTasks.RESLICE.name}")
    if stop_after == PipelineTasks.RESLICE:
        sys.exit()
    ###############################################################################
    #      Reconstruction with either Tomopy-ASTRA (2D) or ToMoBAR-ASTRA (3D)
    with annotate(PipelineTasks.RECONSTRUCT.name, color="blue"):
        recon = reconstruct(data, angles, rot_center, gpu_id)
    print(f"{proc_id} Finished {PipelineTasks.RECONSTRUCT.name}")
    if stop_after == PipelineTasks.RECONSTRUCT:
        sys.exit()
    ###############################################################################
    #                   Saving the result of the reconstruction
    with annotate(PipelineTasks.SAVE.name, color="blue"):
        save_data(recon, run_out_dir, 'reconstruction', comm)
    print(f"{proc_id} Finished {PipelineTasks.SAVE.name}")
