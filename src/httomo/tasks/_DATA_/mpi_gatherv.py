from math import prod
from typing import Optional

import cupy
import numpy
from mpi4py.MPI import Comm


def to_device(
    data: numpy.ndarray, concat_dim: int, comm: Comm, device_rank: int = 0
) -> Optional[cupy.ndarray]:
    ranks_num_slices = comm.gather(data.shape[concat_dim])
    device_data = (
        cupy.empty(
            tuple(
                sum(ranks_num_slices) if dim == concat_dim else dim_size
                for dim, dim_size in enumerate(data.shape)
            ),
            dtype=data.dtype,
        )
        if comm.rank == device_rank
        else None
    )
    comm.Gatherv(
        data,
        (
            device_data,
            [
                prod(
                    rank_num_slices if dim == concat_dim else dim_size
                    for dim, dim_size in enumerate(data.shape)
                )
                for rank_num_slices in ranks_num_slices
            ],
        )
        if device_data is not None
        else None,
        device_rank,
    )
    return device_data


def to_device_inplace(
    data: numpy.ndarray,
    device_data: Optional[cupy.ndarray],
    concat_dim: int,
    comm: Comm,
    device_rank: int = 0,
) -> None:
    ranks_num_slices = comm.gather(data.shape[concat_dim])
    comm.Gatherv(
        data,
        (
            device_data,
            [
                prod(
                    rank_num_slices if dim == concat_dim else dim_size
                    for dim, dim_size in enumerate(data.shape)
                )
                for rank_num_slices in ranks_num_slices
            ],
        )
        if device_data is not None
        else None,
        device_rank,
    )
