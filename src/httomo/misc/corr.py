import numpy as np
from numpy import ndarray
from mpi4py.MPI import Comm

from httomo._stats.globals import max_min_mean_std


def _stats_global_calc(data, flats, darks, mu_dezinger, comm):
    """global stats pre-calculation
    """
    if mu_dezinger is not None and mu_dezinger > 0.0 :
        maxval, minval, mean, std_var_data = max_min_mean_std(data, comm=comm) # for data 
        maxval, minval, mean, std_var_flats = max_min_mean_std(flats, comm=comm) # for flats
        maxval, minval, mean, std_var_darks = max_min_mean_std(darks, comm=comm) # for darks
        std_all_data = (mu_dezinger*std_var_data, mu_dezinger*std_var_flats, mu_dezinger*std_var_darks)
    else:
        std_all_data = (1, 1, 1)
    return std_all_data

def larix(data: ndarray, method_name: str, flats: ndarray, darks: ndarray,
          radius_kernel: int, mu_dezinger: float, ncores: int, comm: Comm):
    """Wrapper for the median and dezinger filtration methods in the larix
    library.

    Args:
        data: A numpy array containing raw projections.
        flats: A numpy array containing the flatfield projections.
        darks: A numpy array containing the dark projections.        
        radius_kernel: a radius of the median kernel (the full size 3D kernel is (2*radius_kernel+1)^3)       
        mu_dezinger: a dezinging parameter, when 0.0 - median filter, when > 0.0 - dezinger
        ncores: The number of CPU cores per process
        comm: The MPI communicator to use.

    Returns:
        tuple[ndarray, ndarray, ndarray]: A tuple of numpy arrays containing the
            filtered projections, flatfields and darkfields.
    """
    from larix.methods.misc import MEDIAN_FILT, MEDIAN_DEZING
    
    # global stats pre-calculation 
    std_all_data = _stats_global_calc(data, flats, darks, mu_dezinger, comm)
        
    if mu_dezinger > 0.0:
        data = MEDIAN_DEZING(data, radius_kernel, std_all_data[0], ncores)
    else:
        data = MEDIAN_FILT(data, radius_kernel, ncores)
    if flats is not None:
        if mu_dezinger > 0.0:
            flats = MEDIAN_DEZING(flats, radius_kernel, std_all_data[1], ncores)            
        else:
            flats = MEDIAN_FILT(flats, radius_kernel, ncores)            
    if darks is not None:
        if mu_dezinger > 0.0:
            darks = MEDIAN_DEZING(darks, radius_kernel, std_all_data[2], ncores)            
        else:            
            darks = MEDIAN_FILT(darks, radius_kernel, ncores)

    return data, flats, darks
