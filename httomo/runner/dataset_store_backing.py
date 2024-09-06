from typing import Tuple

from mpi4py import MPI

from httomo.utils import make_3d_shape_from_shape


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
