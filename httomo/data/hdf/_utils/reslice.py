import logging
from typing import Tuple

import numpy
from mpi4py.MPI import Comm

from httomo.data.mpiutil import alltoall, alltoall_ring
from httomo.data.hdf._utils import chunk
from httomo.utils import log_once


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
        "<-------Reslicing/rechunking the data-------->",
        level=logging.DEBUG,
    )

    rank = comm.rank
    mem_start = _get_memory_usage_mb()

    # No need to reclice anything if there is only one process
    if comm.size == 1:
        log_once(
            f"[Rank {rank}] reslice: Not necessary, as there is only one process",
            level=logging.DEBUG,
        )
        return data, next_slice_dim, 0

    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)
    nprocs = comm.size

    # Calculate split indices
    length = data_shape[next_slice_dim - 1]
    split_indices = [round((length / nprocs) * r) for r in range(nprocs + 1)]

    # Prepare list for alltoall
    to_scatter = []
    total_scatter_size_mb = 0
    for i in range(nprocs):
        start = split_indices[i]
        end = split_indices[i + 1]
        # Use slicing instead of split to avoid intermediate array
        sliced = numpy.take(data, range(start, end), axis=next_slice_dim - 1)
        slice_size_mb = _get_array_size_mb(sliced)
        total_scatter_size_mb += slice_size_mb
        to_scatter.append(sliced)

    # Free original data if possible
    del data

    # All-to-all communication with direct concatenation
    # The concat_axis is current_slice_dim - 1 because we're concatenating along the current slice dimension
    concat_axis = current_slice_dim - 1
    new_data = alltoall_ring(to_scatter, comm, concat_axis=concat_axis)

    # Free scatter list
    del to_scatter

    start_idx = split_indices[rank]
    return new_data, next_slice_dim, start_idx