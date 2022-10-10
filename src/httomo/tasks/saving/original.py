from pathlib import Path

import numpy
from mpi4py.MPI import Comm

from httomo.h5_utils.chunk_h5 import save_dataset


def save_data(data: numpy.ndarray, run_out_dir: Path, comm: Comm) -> None:
    """Write the data to an hdf5 file.

    Args:
        data: The data to be written.
        run_out_dir: The directory to write the file to.
        comm: The MPI communicator to use.
    """
    (vert_slices, recon_x, recon_y) = data.shape
    chunks_recon = (1, recon_x, recon_y)
    save_dataset(run_out_dir, "reconstruction.h5", data, 1, chunks_recon, comm=comm)
