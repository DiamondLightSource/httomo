from pathlib import Path

import numpy
from mpi4py.MPI import Comm

from httomo.h5_utils import chunk_h5, load_h5


def reslice(
    data: numpy.ndarray,
    run_out_dir: Path,
    dimension: int,
    num_angles: int,
    detector_y: int,
    detector_x: int,
    comm: Comm,
) -> tuple[numpy.ndarray, int]:
    """Reslice data by writing to hdf5 store and reading back.

    Args:
        data: The data to be re-sliced.
        run_out_dir: The output directory to write the hdf5 file to.
        dimension: The dimension along which the data is currently sliced.
        angles_total: The total number of slices.
        detector_y: The detector height.
        detector_x: The detector width.
        comm: The MPI communicator to be used.

    Returns:
        tuple[numpy.ndarray, int]: A tuple containing the resliced data and the
            dimension along which it is now sliced.
    """
    # calculate the chunk size for the projection data
    slices_no_in_chunks = 4
    if dimension == 1:
        chunks_data = (slices_no_in_chunks, detector_y, detector_x)
    elif dimension == 2:
        chunks_data = (num_angles, slices_no_in_chunks, detector_x)
    else:
        chunks_data = (num_angles, detector_y, slices_no_in_chunks)

    if dimension == 1:
        chunk_h5.save_dataset(
            run_out_dir,
            "intermediate.h5",
            data,
            dimension,
            chunks_data,
            comm=comm,
        )
        dimension = 2  # assuming sinogram slicing here to get it loaded
        data = load_h5.load_data(
            f"{run_out_dir}/intermediate.h5", dimension, "/data", comm=comm
        )

    return data, dimension
