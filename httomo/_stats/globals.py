from mpi4py import MPI
from numpy import ndarray
import numpy as np
from mpi4py.MPI import Comm


def min_max_mean_std(data: ndarray, comm: Comm):
    """
    global min, max, mean and std_var of the given array

    Args:
        data: A numpy array.
        comm: The MPI communicator to use.
    Returns:
        tuple[(float, float, float, float)]: (min, max, mean, std_var)
    """
    comm.Barrier()
    data = data.flatten()
    # max/min
    maxval_glob = comm.allreduce(data.max(), op=MPI.MAX)
    minval_glob = comm.allreduce(data.min(), op=MPI.MIN)
    # calculating mean
    csum = comm.allreduce(data.sum())
    csize = comm.allreduce(data.size)
    cmean_glob = csum / csize
    # std dev
    rsum = comm.allreduce((abs(data - cmean_glob) ** 2).sum())
    sigma_glob = (rsum / csize) ** 0.5
    return (minval_glob, maxval_glob, cmean_glob, sigma_glob)
