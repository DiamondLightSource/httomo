from numpy import ndarray
from tomopy.prep.stripe import *

def remove_stripes_tomopy(data: ndarray, method_name: str, ncores: int) -> ndarray:
    """Removes stripes with tomopy's tomopy.prep.stripe  methods.

    Args:
        data: A numpy array of projections.
        method_name: the name of method as in tomopy.prep.stripe 
        ncores: The number of CPU cores per process   

    Returns:
        ndarray: A numpy array of projections with stripes removed.
    """
    data = globals()[method_name](data, ncore=ncores)
    return data
