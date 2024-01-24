from typing import Tuple
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Comm

__all__ = [
    "calculate_stats",
]

def calculate_stats(data: np.ndarray,) -> Tuple[float, float, float, int]:
    """Calculating the statistics of the given array
    
    Args:
        data: (np.ndarray): a numpy array
        
    Returns:
        tuple[(float, float, float, int)]: (min, max, sum, total_elements)
    """
    
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)
       
    return (np.min(data), np.max(data), np.sum(data), data.size)
