from typing import Tuple
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm

__all__ = [
    "calculate_stats",
]

def calculate_stats(data: np.ndarray, comm: Comm) -> Tuple[float, float, float, int]:
    """Calculating the global statistics of the given array
    
    Args:
        data: (np.ndarray): a numpy array
        comm: The MPI communicator to use.
        
    Returns:
        tuple[(float, float, float, int)]: (min, max, mean, total_elements)
    """
    
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)
       
    # max/min global values
    minval_glob = comm.allreduce(np.min(data.ravel()), op=MPI.MIN)
    maxval_glob = comm.allreduce(np.max(data.ravel()), op=MPI.MAX)

    # mean calculation
    buf = np.zeros_like(data.ravel())
    MPI.COMM_WORLD.Allreduce(data.ravel(), buf, op=MPI.SUM)
    # the total number of elements in the global data 
    total_elements = MPI.COMM_WORLD.Get_size()
    buf /= MPI.COMM_WORLD.Get_size()
    cmean_glob = np.mean(buf)  
  
    return (minval_glob, maxval_glob, cmean_glob, total_elements)
