import logging
from typing import Tuple
import psutil
import os

import numpy
from mpi4py.MPI import Comm

from httomo.data.mpiutil import alltoall, alltoall_ring
from httomo.data.hdf._utils import chunk
from httomo.utils import log_once


def _get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def _get_array_size_mb(arr):
    """Get array size in MB."""
    return arr.nbytes / 1024 / 1024


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

    log_once(
        f"[Rank {rank}] reslice: Starting memory usage: {mem_start:.2f} MB",
        level=logging.DEBUG,
    )

    # No need to reclice anything if there is only one process
    if comm.size == 1:
        log_once(
            f"[Rank {rank}] reslice: Not necessary, as there is only one process",
            level=logging.DEBUG,
        )
        return data, next_slice_dim, 0

    data_shape = chunk.get_data_shape(data, current_slice_dim - 1)
    nprocs = comm.size

    data_size_mb = _get_array_size_mb(data)
    log_once(
        f"[Rank {rank}] reslice: Input data shape={data.shape}, size={data_size_mb:.2f} MB, dtype={data.dtype}",
        level=logging.DEBUG,
    )

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
    log_once(
        f"[Rank {rank}] reslice: Preparing {nprocs} slices to scatter",
        level=logging.DEBUG,
    )

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
        log_once(
            f"[Rank {rank}] reslice: Prepared slice {i} for rank {i}, shape={sliced.shape}, size={slice_size_mb:.2f} MB",
            level=logging.DEBUG,
        )

    mem_after_scatter = _get_memory_usage_mb()
    log_once(
        f"[Rank {rank}] reslice: to_scatter list created, total size={total_scatter_size_mb:.2f} MB, "
        f"memory usage={mem_after_scatter:.2f} MB (delta: +{mem_after_scatter - mem_start:.2f} MB)",
        level=logging.DEBUG,
    )

    log_once(
        f"[Rank {rank}] reslice: Freeing original data array (size={data_size_mb:.2f} MB)",
        level=logging.DEBUG,
    )
    # Free original data if possible
    del data

    mem_after_del = _get_memory_usage_mb()
    log_once(
        f"[Rank {rank}] reslice: Original data freed, memory usage={mem_after_del:.2f} MB "
        f"(delta: {mem_after_del - mem_after_scatter:.2f} MB)",
        level=logging.DEBUG,
    )

    # All-to-all communication with direct concatenation
    # The concat_axis is current_slice_dim - 1 because we're concatenating along the current slice dimension
    concat_axis = current_slice_dim - 1

    log_once(
        f"[Rank {rank}] reslice: Starting alltoall_ring communication with concat_axis={concat_axis}, "
        f"memory before MPI={mem_after_del:.2f} MB",
        level=logging.DEBUG,
    )

    mem_before_mpi = _get_memory_usage_mb()
    new_data = alltoall_ring(to_scatter, comm, concat_axis=concat_axis)
    mem_after_mpi = _get_memory_usage_mb()

    new_data_size_mb = _get_array_size_mb(new_data)
    log_once(
        f"[Rank {rank}] reslice: alltoall_ring completed, received concatenated array "
        f"shape={new_data.shape}, size={new_data_size_mb:.2f} MB, "
        f"memory usage={mem_after_mpi:.2f} MB (delta: +{mem_after_mpi - mem_before_mpi:.2f} MB)",
        level=logging.DEBUG,
    )

    # Free scatter list
    log_once(
        f"[Rank {rank}] reslice: Freeing to_scatter list (size={total_scatter_size_mb:.2f} MB)",
        level=logging.DEBUG,
    )
    del to_scatter

    mem_after_free = _get_memory_usage_mb()
    log_once(
        f"[Rank {rank}] reslice: to_scatter freed, memory usage={mem_after_free:.2f} MB "
        f"(delta: {mem_after_free - mem_after_mpi:.2f} MB)",
        level=logging.DEBUG,
    )

    start_idx = split_indices[rank]

    mem_end = _get_memory_usage_mb()
    log_once(
        f"[Rank {rank}] reslice: Completed, returning data with slice_dim={next_slice_dim}, "
        f"start_idx={start_idx}, final memory={mem_end:.2f} MB "
        f"(total delta: {mem_end - mem_start:.2f} MB, peak increase: {max(mem_after_scatter, mem_after_mpi) - mem_start:.2f} MB)",
        level=logging.DEBUG,
    )

    return new_data, next_slice_dim, start_idx