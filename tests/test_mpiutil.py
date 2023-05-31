import numpy as np
import pytest
from mpi4py import MPI

from httomo.data import mpiutil


@pytest.mark.mpi
def test_global_ranks():
    assert mpiutil.rank == MPI.COMM_WORLD.rank
    assert mpiutil.size == MPI.COMM_WORLD.size


@pytest.mark.mpi
def test_local_ranks():
    # build reference local rank using MPI hostname
    comm = MPI.COMM_WORLD
    hosts_ranks = {}
    host = MPI.Get_processor_name()
    rank_host = {}
    ret = comm.gather({comm.rank: host})
    if comm.rank == 0:
        for d in ret:
            rank_host.update(d)
    for k, v in rank_host.items():
        if v not in hosts_ranks:
            hosts_ranks[v] = [k]
        else:
            hosts_ranks[v].append(k)

    hosts_ranks = comm.bcast(hosts_ranks)
    rank_local = hosts_ranks[host].index(comm.rank)
    size_local = len(hosts_ranks[host])
    del rank_host

    assert mpiutil.local_rank == rank_local
    assert mpiutil.local_size == size_local


@pytest.mark.mpi
def test_all_to_all():
    data = [
        np.ones((5, 5, 5), dtype=np.uint16) * mpiutil.rank for _ in range(mpiutil.size)
    ]
    rec = mpiutil.alltoall(data)

    expected = [np.ones((5, 5, 5), dtype=np.uint16) * r for r in range(mpiutil.size)]
    np.testing.assert_array_equal(expected, rec)
