from typing import Tuple

import numpy as np
from numpy.typing import DTypeLike
from mpi4py import MPI

from httomo.runner.section import Section
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
