import numpy as np
import pytest
from mpi4py import MPI

from httomo.data.mpiutil import alltoall


@pytest.mark.mpi
def test_all_to_all():
    comm = MPI.COMM_WORLD
    data = [np.ones((5, 5, 5), dtype=np.uint16) * comm.rank for _ in range(comm.size)]
    rec = alltoall(data, comm)

    expected = [np.ones((5, 5, 5), dtype=np.uint16) * r for r in range(comm.size)]
    np.testing.assert_array_equal(expected, rec)
