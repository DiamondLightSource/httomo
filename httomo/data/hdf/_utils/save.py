from pathlib import Path

import numpy
from mpi4py.MPI import Comm

from httomo.utils import print_once
from httomo.data.hdf._utils.chunk import save_dataset


def intermediate_dataset(
    data: numpy.ndarray,
    run_out_dir: Path,
    comm: Comm,
    task_no: int,
    package_name: str,
    method_name: str,
    dataset_name: str,
    recon_algorithm: str = None,
) -> None:
    """Save an intermediate dataset as an hdf file.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be written.
    run_out_dir : Path
        The directory to write the file to.
    comm : Comm
        The MPI communicator to use.
    task_no : int
        The number of the task within the processing pipeline that the given
        dataset is an output of.
    package_name: str
        The package that the method function used came from (ie, "httomo",
        "tomopy", etc.)
    method_name : str
        The method that was used in `package_name` to produce the given dataset.
    dataset_name : str
        The name of the output dataset given in the YAML config.
    recon_algorithm : str
        If the dataset contains a reconstructions, this is the reconstruction
        algorithm name that was used.
    """
    (vert_slices, recon_x, recon_y) = data.shape
    chunks_recon = (1, recon_x, recon_y)

    filename = f"{task_no}-{package_name}-{method_name}-{dataset_name}"
    if recon_algorithm is not None:
        filename = f"{filename}-{recon_algorithm}.h5"
    else:
        filename = f"{filename}.h5"

    print_once(f"Saving intermediate file: {filename}", comm)
    save_dataset(run_out_dir, filename, data, 1, chunks_recon, comm=comm)
