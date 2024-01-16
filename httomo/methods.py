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
    sum_glob = comm.allreduce(np.sum(data.ravel()), op=MPI.SUM)

    # mean calculation
    elem_per_process = int(np.prod(np.shape(data)))
    # the total number of elements in the global data 
    total_elements = MPI.COMM_WORLD.Get_size()*elem_per_process   
    cmean_glob =  sum_glob/total_elements
  
    return (minval_glob, maxval_glob, cmean_glob, total_elements)
