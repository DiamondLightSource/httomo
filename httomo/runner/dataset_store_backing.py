from enum import Enum
from typing import List, Tuple

import numpy as np
from numpy.typing import DTypeLike
from mpi4py import MPI

from httomo.data.hdf._utils.reslice import AllGatherFunc, reslice_memory_estimator
from httomo.runner.section import Section, determine_section_padding
from httomo.utils import _get_slicing_dim, make_3d_shape_from_shape


def calculate_section_input_chunk_shape(
    nprocs: int,
    rank: int,
    global_shape: Tuple[int, int, int],
    slicing_dim: int,
    padding: Tuple[int, int],
) -> Tuple[int, int, int]:
    """
    Calculate the shape of the section input chunk w/ or w/o padding.
    """
    start = round((global_shape[slicing_dim] / nprocs) * rank)
    stop = round((global_shape[slicing_dim] / nprocs) * (rank + 1))
    section_slicing_dim_len = stop - start
    shape = list(global_shape)
    shape[slicing_dim] = section_slicing_dim_len + padding[0] + padding[1]
    return make_3d_shape_from_shape(shape)


def calculate_section_output_chunk_shape(
    chunk_shape: Tuple[int, int, int],
    section: Section,
) -> Tuple[int, int, int]:
    """
    Calculate the shape of the section output chunk that is written to the store. This
    accounts for the data's non-slicing dims changing during processing, which changes the
    chunk shape for the section.
    """
    slicing_dim = _get_slicing_dim(section.pattern) - 1
    non_slice_dims_list = list(chunk_shape)
    non_slice_dims_list.pop(slicing_dim)
    input_non_slice_dims = (non_slice_dims_list[0], non_slice_dims_list[1])
    output_non_slice_dims = input_non_slice_dims

    for method in section.methods:
        if method.memory_gpu is None:
            continue
        output_non_slice_dims = method.calculate_output_dims(input_non_slice_dims)

    output_chunk_shape = list(output_non_slice_dims)
    output_chunk_shape.insert(slicing_dim, chunk_shape[slicing_dim])

    return make_3d_shape_from_shape(output_chunk_shape)


class DataSetStoreBacking(Enum):
    RAM = 1
    File = 2


def estimate_section_memory(
    nprocs: int,
    rank: int,
    allgather_func: AllGatherFunc,
    dtype: DTypeLike,
    global_shape: Tuple[int, int, int],
    sections: List[Section],
    section_idx: int,
) -> int:
    # Get chunk shape created by reader of section `n` (the current section) that will account
    # for padding. This chunk shape is based on the chunk shape written by the writer of
    # section `n - 1` (the previous section)
    padded_input_chunk_shape = calculate_section_input_chunk_shape(
        nprocs=nprocs,
        rank=rank,
        global_shape=global_shape,
        slicing_dim=_get_slicing_dim(sections[section_idx].pattern) - 1,
        padding=determine_section_padding(sections[section_idx]),
    )
    padded_input_chunk_bytes = int(
        np.prod(padded_input_chunk_shape) * np.dtype(dtype).itemsize
    )

    # Get unpadded chunk shape input to current section (for calculation of bytes in output
    # chunk for the current section)
    input_chunk_shape = calculate_section_input_chunk_shape(
        nprocs=nprocs,
        rank=rank,
        global_shape=global_shape,
        slicing_dim=_get_slicing_dim(sections[section_idx].pattern) - 1,
        padding=(0, 0),
    )

    # Get the number of bytes in the input chunk to the section w/ potential modifications to
    # the non-slicing dims, to then determine the number of bytes in the output chunk written
    # by the current section
    output_chunk_shape = calculate_section_output_chunk_shape(
        chunk_shape=input_chunk_shape,
        section=sections[section_idx],
    )
    output_chunk_bytes = int(np.prod(output_chunk_shape) * np.dtype(dtype).itemsize)

    # If a reslice operation would occur in moving from the current section to the next
    # section, then calculate the number of bytes the reslice operation would take, given the
    # input to it (which would be the output chunk of the current section)
    reslice_bytes = 0
    if (
        nprocs > 1
        and section_idx < len(sections) - 1
        and sections[section_idx].pattern != sections[section_idx + 1].pattern
    ):
        ring_algorithm_bytes, reslice_output_bytes = reslice_memory_estimator(
            output_chunk_shape,
            dtype,
            _get_slicing_dim(sections[section_idx].pattern),
            _get_slicing_dim(sections[section_idx + 1].pattern),
            nprocs,
            rank,
            allgather_func,
        )
        reslice_bytes += ring_algorithm_bytes + reslice_output_bytes

    # TODO: The nature of the pinned memory allocations by cupy is currently under
    # investigation, so a more precise calculation for its size is not yet known.
    #
    # It's known that this can grow quite large via allocations exceeding the current
    # allocation being bumped to the next power of 2 (ie, a 16GiB allocation that is exceeded
    # by 1 byte will have a 32GiB allocation made in addition to the original 16GiB).
    #
    # Taking half the input data size seems to be in the ballpark for what has been observed
    # with larger datasets (ie, an 84GB dataset being processed took ~520GB of memory, and with
    # this arbitrary choice of 0.5 as a multiplicative factor gets the estimated value to
    # ~514GB)
    CUPY_PINNED_CPU_MEMORY = int(0.5 * np.prod(global_shape) * np.dtype(dtype).itemsize)

    return (
        padded_input_chunk_bytes
        + output_chunk_bytes
        + reslice_bytes
        + CUPY_PINNED_CPU_MEMORY
    )


def determine_store_backing(
    comm: MPI.Comm,
    sections: List[Section],
    memory_limit_bytes: int,
    dtype: DTypeLike,
    global_shape: Tuple[int, int, int],
    section_idx: int,
) -> DataSetStoreBacking:
    section_memory = estimate_section_memory(
        comm.size,
        comm.rank,
        comm.allgather,
        dtype,
        global_shape,
        sections,
        section_idx,
    )

    send_buffer = np.zeros(1, dtype=bool)
    recv_buffer = np.zeros(1, dtype=bool)

    if memory_limit_bytes > 0 and section_memory >= memory_limit_bytes:
        send_buffer[0] = True

    # do a logical OR of all the enum variants across the processes
    comm.Allreduce([send_buffer, MPI.BOOL], [recv_buffer, MPI.BOOL], MPI.LOR)

    if bool(recv_buffer[0]) is True:
        return DataSetStoreBacking.File

    return DataSetStoreBacking.RAM
