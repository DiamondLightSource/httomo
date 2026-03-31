import logging
from typing import Tuple

import numpy
from mpi4py.MPI import Comm

from httomo.data.mpiutil import alltoall_ring
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

    # No need to reslice anything if there is only one process
    if comm.size == 1:
        log_once(
            "Reslicing not necessary, as there is only one process",
            level=logging.DEBUG,
        )
        return data, next_slice_dim, 0

    # Get shape of full/unsplit data, in order to set the chunk shape based on
    # the dims of the full data rather than of the split data
    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)

    nprocs = comm.size

    # Calculate split indices
    length = data_shape[next_slice_dim - 1]
    split_indices = [round((length / nprocs) * r) for r in range(1, nprocs)]

    # Prepare list for alltoall_ring
    to_scatter = numpy.split(data, split_indices, axis=next_slice_dim - 1)

    # All-to-all communication with direct concatenation
    new_data = alltoall_ring(to_scatter, comm, concat_axis=current_slice_dim - 1)

    # Free scatter list
    del to_scatter

    start_idx = 0 if comm.rank == 0 else split_indices[comm.rank - 1]
    return new_data, next_slice_dim, start_idx


def reslice_memory_estimator(
    data_shape: Tuple[int, int, int],
    dtype: numpy.dtype,
    current_slice_dim: int,
    next_slice_dim: int,
    comm: Comm,
) -> dict:
    rank = comm.rank
    nprocs = comm.size
    itemsize = numpy.dtype(dtype).itemsize
    input_size = numpy.prod(data_shape) * itemsize

    split_sizes = []
    length = data_shape[next_slice_dim]
    split_indices = [round((length / nprocs) * r) for r in range(1, nprocs)]

    prev_idx = 0
    for i in range(nprocs):
        next_idx = split_indices[i] if i < len(split_indices) else length
        split_shape = list(data_shape)
        split_shape[next_slice_dim] = next_idx - prev_idx
        split_sizes.append(numpy.prod(split_shape) * itemsize)
        prev_idx = next_idx

    total_split_size = sum(split_sizes)

    all_split_sizes = comm.allgather(split_sizes)
    recv_sizes = [all_split_sizes[p][rank] for p in range(nprocs)]

    output_shape = list(data_shape)
    output_shape[current_slice_dim] = sum(
        recv_sizes[p]
        // (
            itemsize
            * numpy.prod([data_shape[d] for d in range(3) if d != next_slice_dim])
        )
        for p in range(nprocs)
    )
    output_size = numpy.prod(output_shape) * itemsize

    max_send_buffer = max(split_sizes)
    max_recv_buffer = max(recv_sizes)

    from httomo.data.mpiutil import _mpi_max_elements

    max_elements = _mpi_max_elements - 1
    max_transfer_elements = max(
        max(split_sizes) // itemsize, max(recv_sizes) // itemsize
    )

    needs_chunking = max_transfer_elements > max_elements

    if needs_chunking:
        chunk_overhead_send = max_send_buffer
        chunk_overhead_recv = max_recv_buffer
    else:
        chunk_overhead_send = 0
        chunk_overhead_recv = 0

    peak_before_ring = input_size + total_split_size + output_size

    peak_during_ring = (
        peak_before_ring
        + max_send_buffer  # Temporary send buffer
        + max_recv_buffer  # Temporary recv buffer
        + chunk_overhead_send  # Flattened send array (if chunking)
        + chunk_overhead_recv  # Flattened recv array (if chunking)
    )

    peak_after_ring = output_size

    return max(peak_before_ring, peak_during_ring, peak_after_ring) * 1.01
