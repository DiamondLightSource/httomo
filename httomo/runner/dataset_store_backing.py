from enum import Enum
from typing import List, Tuple

import numpy as np
from numpy.typing import DTypeLike
from mpi4py import MPI

from httomo.runner.section import Section, determine_section_padding
from httomo.utils import _get_slicing_dim, make_3d_shape_from_shape


def calculate_section_input_chunk_shape(
    comm: MPI.Comm,
    global_shape: Tuple[int, int, int],
    slicing_dim: int,
    padding: Tuple[int, int],
) -> Tuple[int, int, int]:
    """
    Calculate the shape of the section input chunk w/ or w/o padding.
    """
    start = round((global_shape[slicing_dim] / comm.size) * comm.rank)
    stop = round((global_shape[slicing_dim] / comm.size) * (comm.rank + 1))
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


def _non_last_section_in_pipeline(
    memory_limit_bytes: int,
    write_chunk_bytes: int,
    read_chunk_bytes: int,
) -> DataSetStoreBacking:
    """
    Calculate backing of dataset store for non-last sections in pipeline
    """
    if (
        memory_limit_bytes > 0
        and write_chunk_bytes + read_chunk_bytes >= memory_limit_bytes
    ):
        return DataSetStoreBacking.File

    return DataSetStoreBacking.RAM


def determine_store_backing(
    comm: MPI.Comm,
    sections: List[Section],
    memory_limit_bytes: int,
    dtype: DTypeLike,
    global_shape: Tuple[int, int, int],
    section_idx: int,
) -> DataSetStoreBacking:
    # Get chunk shape created by reader of section `n` (the current section) that will account
    # for padding. This chunk shape is based on the chunk shape written by the writer of
    # section `n - 1` (the previous section)
    padded_input_chunk_shape = calculate_section_input_chunk_shape(
        comm=comm,
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
        comm=comm,
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

    send_buffer = np.zeros(1, dtype=bool)
    recv_buffer = np.zeros(1, dtype=bool)
    store_backing = _non_last_section_in_pipeline(
        memory_limit_bytes=memory_limit_bytes,
        read_chunk_bytes=padded_input_chunk_bytes,
        write_chunk_bytes=output_chunk_bytes,
    )

    if store_backing is DataSetStoreBacking.File:
        send_buffer[0] = True

    # do a logical OR of all the enum variants across the processes
    comm.Allreduce([send_buffer, MPI.BOOL], [recv_buffer, MPI.BOOL], MPI.LOR)

    if bool(recv_buffer[0]) is True:
        return DataSetStoreBacking.File

    return DataSetStoreBacking.RAM
