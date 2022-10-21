from pathlib import Path

import numpy
from mpi4py.MPI import Comm

from httomo.data.hdf._utils.chunk import save_dataset


def intermediate_dataset(data: numpy.ndarray, run_out_dir: Path,
                         method_name: str, comm: Comm) -> None:
    """Save an intermeidate dataset as an hdf file.

    Args:
        data: The data to be written.
        run_out_dir: The directory to write the file to.
        method_name: the dataset name
        comm: The MPI communicator to use.
    """
    (vert_slices, recon_x, recon_y) = data.shape
    chunks_recon = (1, recon_x, recon_y)
    save_dataset(run_out_dir, method_name+".h5", data, 1, chunks_recon, comm=comm)
