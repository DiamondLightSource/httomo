from pathlib import Path

import numpy
from mpi4py.MPI import Comm

from httomo.data.hdf._utils import chunk, load
from httomo.utils import print_once

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

    Parameters
    ----------
    data : numpy.ndarray
        The data to be re-sliced.
    run_out_dir : Path
        The output directory to write the hdf5 file to.
    dimension : int
        The dimension along which the data is currently sliced.
    num_angles : int
        The total number of slices.
    detector_y : int
        The detector height.
    detector_x : int
        The detector width.
    comm : Comm
        The MPI communicator to be used.

    Returns:
    tuple[numpy.ndarray, int]:
        A tuple containing the resliced data and the dimension along which it is
        now sliced.
    """
    # calculate the chunk size for the projection data
    slices_no_in_chunks = 1
    if dimension == 1:
        chunks_data = (slices_no_in_chunks, detector_y, detector_x)
    elif dimension == 2:
        chunks_data = (num_angles, slices_no_in_chunks, detector_x)
    else:
        chunks_data = (num_angles, detector_y, slices_no_in_chunks)

    print_once(f"<-------Reslicing/rechunking the data-------->", comm, colour="blue")
    if dimension == 1:
        chunk.save_dataset(
            run_out_dir,
            "intermediate.h5",
            data,
            dimension,
            chunks_data,
            comm=comm,
        )
        dimension = 2  # assuming sinogram slicing here to get it loaded
        data = load.load_data(
            f"{run_out_dir}/intermediate.h5", dimension, "/data", comm=comm
        )

    return data, dimension
