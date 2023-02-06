from pathlib import Path

import numpy
from mpi4py.MPI import Comm

from httomo.data.hdf._utils import chunk, load
from httomo.utils import print_once

def reslice(
    data: numpy.ndarray,
    run_out_dir: Path,
    current_slice_dim: int,
    next_slice_dim: int,
    comm: Comm,
) -> tuple[numpy.ndarray, int]:
    """Reslice data by writing to hdf5 store with data chunked along a different
    dimension, and reading back along the new chunking dimension.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be re-sliced.
    run_out_dir : Path
        The output directory to write the hdf5 file to.
    current_slice_dim : int
        The dimension along which the data is currently sliced.
    next_slice_dim : int
        The dimension along which the data should be sliced after re-chunking
        and saving.
    comm : Comm
        The MPI communicator to be used.

    Returns:
    tuple[numpy.ndarray, int]:
        A tuple containing the resliced data and the dimension along which it is
        now sliced.
    """
    # Get shape of full/unsplit data, in order to set the chunk shape based on
    # the dims of the full data rather than of the split data
    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)

    # Calculate the chunk size for the resliced data
    slices_no_in_chunks = 1
    if next_slice_dim == 1:
        # Chunk along projection (rotation angle) dimension
        chunks_data = (slices_no_in_chunks, data_shape[1], data_shape[2])
    elif next_slice_dim == 2:
        # Chunk along sinogram (detector y) dimension
        chunks_data = (data_shape[0], slices_no_in_chunks, data_shape[2])
    else:
        # Chunk along detector x dimension
        chunks_data = (data_shape[0], data_shape[1], slices_no_in_chunks)

    print_once(f"<-------Reslicing/rechunking the data-------->", comm, colour="blue")
    # Pass the current slicing dim so then data can be gathered and assembled
    # correctly, and the new chunk shape to save the data in an hdf5 file with
    # the new chunking
    chunk.save_dataset(
        run_out_dir,
        "intermediate.h5",
        data,
        current_slice_dim,
        chunks_data,
        reslice=True,
        comm=comm,
    )
    # Read data back along the new slicing dimension
    data = load.load_data(
        f"{run_out_dir}/intermediate.h5", next_slice_dim, "/data", comm=comm
    )

    return data, next_slice_dim
