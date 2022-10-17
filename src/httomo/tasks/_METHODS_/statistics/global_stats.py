from mpi4py import MPI
from numpy import ndarray
import numpy as np
from mpi4py.MPI import Comm


def max_min_mean_std(data: ndarray, comm: Comm):
    """
    global max, min, mean and std_var of the given array

    Args:
        data: A numpy array.
        comm: The MPI communicator to use.
    Returns:
        tuple[float, float, float, float]: max, min, mean, std_var
    """
    comm.Barrier()
    data = data.flatten()
    # max/min
    maxval = comm.allreduce(data.max(), op=MPI.MAX)
    minval = comm.allreduce(data.min(), op=MPI.MIN)
    # calculating mean
    csum = comm.allreduce(data.sum())
    csize = comm.allreduce(data.size)
    cmean = csum / csize
    # std dev
    rsum = comm.allreduce((abs(data - cmean)**2).sum())
    sigma = (rsum / csize)**0.5
    return maxval, minval, cmean, sigma