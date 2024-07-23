import numpy as np
import pytest
from mpi4py import MPI

from httomo.data import mpiutil


@pytest.mark.mpi
def test_global_ranks():
    assert mpiutil.rank == MPI.COMM_WORLD.rank
    assert mpiutil.size == MPI.COMM_WORLD.size


@pytest.mark.mpi
def test_all_to_all():
    data = [
        np.ones((5, 5, 5), dtype=np.uint16) * mpiutil.rank for _ in range(mpiutil.size)
    ]
    rec = mpiutil.alltoall(data)

    expected = [np.ones((5, 5, 5), dtype=np.uint16) * r for r in range(mpiutil.size)]
    np.testing.assert_array_equal(expected, rec)
