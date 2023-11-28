from os import PathLike
from pathlib import Path
from typing import Optional, Tuple
from mpi4py import MPI

import numpy
from mpi4py.MPI import Comm

from httomo.data import mpiutil
from httomo.data.hdf._utils import chunk, load
from httomo.utils import Colour, log_once


def reslice(
    data: numpy.ndarray,
    current_slice_dim: int,
    next_slice_dim: int,
    comm: Comm,
) -> Tuple[numpy.ndarray, int, int]:
    """Reslice data by using in-memory MPI directives.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be re-sliced.
    current_slice_dim : int
        The dimension along which the data is currently sliced.
    next_slice_dim : int
        The dimension along which the data should be sliced after re-chunking
        and saving.
    comm : Comm
        The MPI communicator to be used.

    Returns:
    tuple[numpy.ndarray, int, int]:
        A tuple containing the resliced data, the dimension along which it is
        now sliced, and the starting index in slicing dimension for the current process.
    """
    log_once(
        f"<-------Reslicing/rechunking the data-------->",
        comm,
        colour=Colour.BLUE,
        level=1,
    )

    # No need to reclice anything if there is only one process
    if mpiutil.size == 1:
        log_once(
            "Reslicing not necessary, as there is only one process",
            comm=comm,
            colour=Colour.BLUE,
            level=1,
        )
        return data, next_slice_dim, 0

    # Get shape of full/unsplit data, in order to set the chunk shape based on
    # the dims of the full data rather than of the split data
    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)

    # build a list of what each process has to scatter to others
    nprocs = mpiutil.size
    length = data_shape[next_slice_dim - 1]
    split_indices = [round((length / nprocs) * r) for r in range(1, nprocs)]
    to_scatter = numpy.split(data, split_indices, axis=next_slice_dim - 1)

    # all-to-all MPI call distributes every processes list to every other process,
    # and we concatenate them again across the resliced dimension
    new_data = numpy.concatenate(
        mpiutil.alltoall(to_scatter), axis=current_slice_dim - 1
    )

    start_idx = 0 if comm.rank == 0 else split_indices[comm.rank-1]
    return new_data, next_slice_dim, start_idx


def reslice_filebased(
    data: numpy.ndarray,
    current_slice_dim: int,
    next_slice_dim: int,
    comm: Comm,
    reslice_dir: PathLike,
) -> Tuple[numpy.ndarray, int, int]:
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
    tuple[numpy.ndarray, int, int]:
        A tuple containing the resliced data and the dimension along which it is
        now sliced and the start index in that dimension.
    """
    # Get shape of full/unsplit data, in order to set the chunk shape based on
    # the dims of the full data rather than of the split data
    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)

    # Calculate the chunk size for the resliced data
    slices_no_in_chunks = 1
    chunks_data = list(data_shape)
    chunks_data[next_slice_dim - 1] = slices_no_in_chunks
    
    log_once(
        f"<-------Reslicing/rechunking the data-------->",
        comm,
        colour=Colour.BLUE,
        level=1,
    )
    # Pass the current slicing dim so then data can be gathered and assembled
    # correctly, and the new chunk shape to save the data in an hdf5 file with
    # the new chunking
    chunk.save_dataset(
        reslice_dir,
        "intermediate.h5",
        data,
        slice_dim=current_slice_dim,
        chunks=tuple(chunks_data),
        reslice=True,
        comm=comm,
    )
    # Read data back along the new slicing dimension
    data, start_idx = load.load_data(
        f"{reslice_dir}/intermediate.h5", next_slice_dim, "/data", comm=comm
    )

    return data, next_slice_dim, start_idx


def single_sino_reslice(
    data: numpy.ndarray,
    idx: int,
) -> Optional[numpy.ndarray]:
    if mpiutil.size == 1:
        log_once(
            "Reslicing for single sinogram not necessary, as there is only one process",
            comm=mpiutil.comm,
            colour=Colour.BLUE,
            level=1,
        )
        return data[:, idx, :]

    NUMPY_DTYPE = numpy.float32
    MPI_DTYPE = MPI.FLOAT

    # Get shape of full/unsplit data, in order to define the shape of the numpy
    # array that will hold the gathered data
    data_shape = chunk.get_data_shape(data, 0)

    if mpiutil.rank == 0:
        # Define the numpy array that will hold the single sinogram that has
        # been gathered from data from all MPI processes
        recvbuf = numpy.empty(data_shape[0]*data_shape[2], dtype=NUMPY_DTYPE)
    else:
        recvbuf = None
    # From the full projections that an MPI process has, send the data that
    # contributes to the sinogram at height `idx` (ie, send a "partial
    # sinogram")
    sendbuf = numpy.ascontiguousarray(
        data[:, idx, :].reshape(data[:, idx, :].size), dtype=NUMPY_DTYPE
    )
    sizes_rec = mpiutil.comm.gather(sendbuf.size)
    # Gather the data into the rank 0 process
    mpiutil.comm.Gatherv(
        (sendbuf, data.shape[0]*data.shape[2], MPI_DTYPE),
        (recvbuf, sizes_rec, MPI_DTYPE),
        root=0
    )

    if mpiutil.rank == 0:
        assert recvbuf is not None
        return recvbuf.reshape((data_shape[0], data_shape[2]))
    else:
        return None
