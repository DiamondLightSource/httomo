from os import PathLike
from typing import Optional, Tuple

import h5py as h5
from mpi4py import MPI
from numpy import ndarray


def get_data_shape_and_offset(
    data: ndarray, dim: int, comm: MPI.Comm = MPI.COMM_WORLD
) -> Tuple[Tuple, Tuple]:
    """Gets the shape of the distribtued dataset as well as the offset of the
    local one within the full data shape.

    Parameters
    ----------
    data : ndarray
        The process data.
    dim : int
        The dimension in which to get the shape.
    comm : MPI.Comm
        The MPI communicator.

    Returns
    -------
    Tuple, Tuple
        The shape of the given distributed dataset and the global starting index of this dataset,
        both as a 3-tuple of integers
    """
    shape = list(data.shape)
    lengths = comm.gather(shape[dim], 0)
    lengths = comm.bcast(lengths, 0)
    assert lengths is not None
    assert len(lengths) > 0
    assert isinstance(lengths[0], int)
    shape[dim] = sum(lengths)
    global_index = [0, 0, 0]
    global_index[dim] = sum(lengths[: comm.rank - 1])
    return tuple(shape), tuple(global_index)


def get_data_shape(data: ndarray, dim: int, comm: MPI.Comm = MPI.COMM_WORLD) -> Tuple:
    """Gets the shape of a distributed dataset.

    Parameters
    ----------
    data : ndarray
        The process data.
    dim : int
        The dimension in which to get the shape.
    comm : MPI.Comm
        The MPI communicator.

    Returns
    -------
    Tuple
        The shape of the given distributed dataset.
    """
    return get_data_shape_and_offset(data, dim, comm)[0]
