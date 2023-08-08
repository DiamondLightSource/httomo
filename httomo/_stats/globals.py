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
    
    #comm.Barrier()
    
    # max/min global values
    maxval_glob = comm.allreduce(np.max(data.ravel()), op=MPI.MAX)
    minval_glob = comm.allreduce(np.min(data.ravel()), op=MPI.MIN)
        
    # mean calculation
    buf = np.zeros_like(data.ravel())
    MPI.COMM_WORLD.Allreduce(data.ravel(), buf, op=MPI.SUM)
    buf /= MPI.COMM_WORLD.Get_size()
    cmean_glob = np.mean(buf)       
   
    #TODO: standard deviation, follow
    # https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/
    # rsum = comm.allreduce((abs(data - cmean_glob) ** 2).sum())
    # sigma_glob = (rsum / csize) ** 0.5
    sigma_glob = 0.0
    
    return minval_glob, maxval_glob, cmean_glob, sigma_glob
