import time
from unittest import mock

import numpy as np
import pytest
from mpi4py import MPI

from httomo.data.hdf._utils.reslice import reslice


@pytest.mark.parametrize(
    "current_slice_dim, next_slice_dim",
    [(1, 2), (2, 1), (1, 3), (2, 3), (3, 1), (3, 2)],
    ids=[
        "proj2sino",
        "sino2proj",
        "proj2third",
        "sino2third",
        "third2proj",
        "third2sino",
    ],
)
@pytest.mark.parametrize(
    "full_shape",
    [
        (15, 13, 9),
        (12, 3, 10),
        (1, 4, 12),
        (10, 1, 12),
        (1, 1, 4),
        (4, 5, 1),
        (13, 23, 51),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
@pytest.mark.mpi
def test_reslice(full_shape, current_slice_dim, next_slice_dim, dtype):
    """This test checks the reclice function in all possible dimensions and
    reslicing parameters. It should work without MPI (in which case the
    output data is just the same as the input data) and with MPI for any
    number of processes.

    To run with MPI, run:

    mpirun -np 2 pytest -m mpi

    (the marker makes sure only tests that are relevant for MPI are run)
    """

    comm = MPI.COMM_WORLD

    # every process creates a slice of the full shape in current_slice_dim,
    # and we set all the values to the rank index to ease assertions
    start = round(full_shape[current_slice_dim - 1] / comm.size * comm.rank)
    stop = round(full_shape[current_slice_dim - 1] / comm.size * (comm.rank + 1))
    in_shape = np.copy(full_shape)
    in_shape[current_slice_dim - 1] = stop - start
    data = np.ones(in_shape, dtype=dtype) * comm.rank

    # reslice, artificially mocking MPI max size to check that it can handle sizes
    # larger than max
    with mock.patch("httomo.data.mpiutil._mpi_max_elements", 128):
        newdata, _, start_idx = reslice(data, current_slice_dim, next_slice_dim, comm)

    # check expected dimensions
    start = round((full_shape[next_slice_dim - 1] / comm.size) * comm.rank)
    stop = round((full_shape[next_slice_dim - 1] / comm.size) * (comm.rank + 1))
    expected_dims = np.copy(full_shape)
    expected_dims[next_slice_dim - 1] = stop - start
    np.testing.assert_array_equal(newdata.shape, expected_dims)
    assert start == start_idx

    # check expected data
    expected = np.ones(expected_dims, dtype=dtype)
    for r in range(comm.size):
        start = round(full_shape[current_slice_dim - 1] / comm.size * r)
        stop = round(full_shape[current_slice_dim - 1] / comm.size * (r + 1))
        if current_slice_dim == 2:
            expected[:, start:stop, :] *= r
        elif current_slice_dim == 1:
            expected[start:stop, :, :] *= r
        elif current_slice_dim == 3:
            expected[:, :, start:stop] *= r
    np.testing.assert_array_equal(expected, newdata)


@pytest.mark.mpi
@pytest.mark.perf
def test_reslice_performance():
    comm = MPI.COMM_WORLD

    process_shape = (1801, 15, 2560)
    current_slice_dim = 1
    next_slice_dim = 2
    data = np.ones(process_shape, dtype=np.float32) * comm.rank

    # reslice
    start = time.perf_counter_ns()
    for _ in range(10):
        reslice(data, current_slice_dim, next_slice_dim, comm)

    duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

    duration_ms = comm.reduce(duration_ms, MPI.MAX)

    if comm.rank == 0:
        assert "performance in ms" == duration_ms
