from enum import Enum
from typing import Callable, List, ParamSpec, Tuple

import numpy as np
from numpy.typing import DTypeLike
from mpi4py import MPI

from httomo.runner.section import Section, determine_section_padding
from httomo.utils import _get_slicing_dim, make_3d_shape_from_shape


def calculate_section_chunk_shape(
    comm: MPI.Comm,
    global_shape: Tuple[int, int, int],
    slicing_dim: int,
    padding: Tuple[int, int],
) -> Tuple[int, int, int]:
    """
    Calculate chunk shape (w/ or w/o padding) for a section.
    """
    start = round((global_shape[slicing_dim] / comm.size) * comm.rank)
    stop = round((global_shape[slicing_dim] / comm.size) * (comm.rank + 1))
    section_slicing_dim_len = stop - start
    shape = list(global_shape)
    shape[slicing_dim] = section_slicing_dim_len + padding[0] + padding[1]
    return make_3d_shape_from_shape(shape)


def calculate_section_chunk_bytes(
    chunk_shape: Tuple[int, int, int],
    dtype: DTypeLike,
    section: Section,
) -> int:
    """
    Calculate the number of bytes in the section output chunk that is written to the store. Ths
    accounts for data's non-slicing dims changing during processing, which changes the chunk
    shape for the section and thus affects the number of bytes in the chunk.
    """
    slicing_dim = _get_slicing_dim(section.pattern) - 1
    non_slice_dims_list = list(chunk_shape)
    non_slice_dims_list.pop(slicing_dim)
    non_slice_dims = (non_slice_dims_list[0], non_slice_dims_list[1])

    for method in section.methods:
        if method.memory_gpu is None:
            continue
        non_slice_dims = method.calculate_output_dims(non_slice_dims)

    return int(
        np.prod(non_slice_dims) * chunk_shape[slicing_dim] * np.dtype(dtype).itemsize
    )


class DataSetStoreBacking(Enum):
    RAM = 1
    File = 2


P = ParamSpec("P")


def _reduce_decorator_factory(
    comm: MPI.Comm,
) -> Callable[[Callable[P, DataSetStoreBacking]], Callable[P, DataSetStoreBacking]]:
    """
    Generate decorator for store-backing calculator function that will use the given MPI
    communicator for the reduce operation.
    """

    def reduce_decorator(
        func: Callable[P, DataSetStoreBacking]
    ) -> Callable[P, DataSetStoreBacking]:
        """
        Decorator for store-backing calculator function.
        """

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> DataSetStoreBacking:
            """
            Perform store-backing calculation across all MPI processes and reduce.
            """
            # reduce store backing enum variant across all processes - if any has
            # `File` variant, all should use a file
            send_buffer = np.zeros(1, dtype=bool)
            recv_buffer = np.zeros(1, dtype=bool)
            store_backing = func(*args, **kwargs)

            if store_backing is DataSetStoreBacking.File:
                send_buffer[0] = True

            # do a logical or of all the enum variants across the processes
            comm.Allreduce([send_buffer, MPI.BOOL], [recv_buffer, MPI.BOOL], MPI.LOR)

            if bool(recv_buffer[0]) is True:
                return DataSetStoreBacking.File

            return DataSetStoreBacking.RAM

        return wrapper

    return reduce_decorator


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


def _last_section_in_pipeline(
    memory_limit_bytes: int,
    write_chunk_bytes: int,
) -> DataSetStoreBacking:
    """
    Calculate backing of dataset store for last section in pipeline
    """
    if memory_limit_bytes > 0 and write_chunk_bytes >= memory_limit_bytes:
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
    reduce_decorator = _reduce_decorator_factory(comm)

    # Get chunk shape input to section
    current_chunk_shape = calculate_section_chunk_shape(
        comm=comm,
        global_shape=global_shape,
        slicing_dim=_get_slicing_dim(sections[section_idx].pattern) - 1,
        padding=(0, 0),
    )

    # Get the number of bytes in the input chunk to the section w/ potential modifications to
    # the non-slicing dims
    current_chunk_bytes = calculate_section_chunk_bytes(
        chunk_shape=current_chunk_shape,
        dtype=dtype,
        section=sections[section_idx],
    )

    if section_idx == len(sections) - 1:
        return reduce_decorator(_last_section_in_pipeline)(
            memory_limit_bytes=memory_limit_bytes,
            write_chunk_bytes=current_chunk_bytes,
        )

    # Get chunk shape created by reader of section `n+1`, that will add padding to the
    # chunk shape written by the writer of section `n`
    next_chunk_shape = calculate_section_chunk_shape(
        comm=comm,
        global_shape=global_shape,
        slicing_dim=_get_slicing_dim(sections[section_idx + 1].pattern) - 1,
        padding=determine_section_padding(sections[section_idx + 1]),
    )
    next_chunk_bytes = int(np.prod(next_chunk_shape) * np.dtype(dtype).itemsize)
    return reduce_decorator(_non_last_section_in_pipeline)(
        memory_limit_bytes=memory_limit_bytes,
        write_chunk_bytes=current_chunk_bytes,
        read_chunk_bytes=next_chunk_bytes,
    )
