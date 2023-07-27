import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm


def min_max_mean_std(data: np.ndarray, comm: Comm):
    """calculating global statistics of the given array
    Args:
        data: (np.ndarray): a numpy array
        comm: The MPI communicator to use.
    Returns:
        tuple[(float, float, float, float)]: (min, max, mean, std_var)
    """

    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)
    data = data.flatten()
    
    minval_glob = 0.0
    maxval_glob = 0.0
    cmean_glob = 0.0
    sigma_glob = 0.0
    
    if comm.rank == 0:
        comm.Barrier()
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
