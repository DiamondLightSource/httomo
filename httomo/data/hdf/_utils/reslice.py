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

    # No need to reclice anything if there is only one process
    if comm.size == 1:
        log_once(
            "Reslicing not necessary, as there is only one process",
            level=logging.DEBUG,
        )
        return data, next_slice_dim, 0

    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)
    nprocs = comm.size
    rank = comm.rank

    # Calculate split indices
    length = data_shape[next_slice_dim - 1]
    split_indices = [round((length / nprocs) * r) for r in range(nprocs + 1)]

    log_once(
        f"[Rank {rank}] reslice: Splitting dimension {next_slice_dim} of length {length} into {nprocs} parts",
        level=logging.DEBUG,
    )
    log_once(
        f"[Rank {rank}] reslice: Split indices: {split_indices}",
        level=logging.DEBUG,
    )

    # Prepare list for alltoall
    to_scatter = []
    for i in range(nprocs):
        start = split_indices[i]
        end = split_indices[i + 1]
        # Use slicing instead of split to avoid intermediate array
        sliced = numpy.take(data, range(start, end), axis=next_slice_dim - 1)
        to_scatter.append(sliced)
        log_once(
            f"[Rank {rank}] reslice: Prepared slice for rank {i}, shape={sliced.shape}",
            level=logging.DEBUG,
        )

    log_once(
        f"[Rank {rank}] reslice: Freeing original data array before MPI communication",
        level=logging.DEBUG,
    )
    # Free original data if possible
    del data

    # All-to-all communication
    log_once(
        f"[Rank {rank}] reslice: Starting alltoall_ring communication",
        level=logging.DEBUG,
    )
    received = alltoall_ring(to_scatter, comm)

    log_once(
        f"[Rank {rank}] reslice: alltoall_ring completed, received {len(received)} arrays",
        level=logging.DEBUG,
    )
    log_once(
        f"[Rank {rank}] reslice: Received array shapes: {[arr.shape for arr in received]}",
        level=logging.DEBUG,
    )

    # Free scatter list
    del to_scatter

    # Calculate output shape and pre-allocate to avoid concatenate memory issues
    concat_axis = current_slice_dim - 1
    output_shape = list(received[0].shape)
    output_shape[concat_axis] = sum(arr.shape[concat_axis] for arr in received)

    log_once(
        f"[Rank {rank}] reslice: Pre-allocating output array with shape {tuple(output_shape)}, "
        f"dtype={received[0].dtype}",
        level=logging.DEBUG,
    )

    # Pre-allocate the output array
    new_data = numpy.empty(output_shape, dtype=received[0].dtype)

    # Copy received arrays into the output array piece by piece
    current_pos = 0
    for i, arr in enumerate(received):
        size_along_axis = arr.shape[concat_axis]

        # Create slice objects for the copy operation
        slices = [slice(None)] * arr.ndim
        slices[concat_axis] = slice(current_pos, current_pos + size_along_axis)

        log_once(
            f"[Rank {rank}] reslice: Copying array {i} into output at position {current_pos}:{current_pos + size_along_axis}",
            level=logging.DEBUG,
        )

        new_data[tuple(slices)] = arr
        current_pos += size_along_axis

        # Free the received array immediately after copying
        del arr

    # Free the received list
    del received

    log_once(
        f"[Rank {rank}] reslice: Memory-efficient concatenation completed, final shape={new_data.shape}",
        level=logging.DEBUG,
    )

    start_idx = split_indices[rank]

    log_once(
        f"[Rank {rank}] reslice: Completed, returning data with slice_dim={next_slice_dim}, start_idx={start_idx}",
        level=logging.DEBUG,
    )

    return new_data, next_slice_dim, start_idx