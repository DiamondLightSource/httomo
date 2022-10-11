from mpi4py.MPI import Comm
from numpy import ndarray
import numpy as np


def median3d_larix(data: ndarray, flats: ndarray, darks: ndarray, radius_kernel: int, mu_dezinger: float, ncores: int):
    """Performs median or dezinger filtration on the data using the larix library.

    Args:
        data: A numpy array containing raw projections.
        flats: A numpy array containing the flatfield projections.
        darks: A numpy array containing the dark projections.        
        radius_kernel: a radius of the median kernel (the full size 3D kernel is (2*radius_kernel+1)^3)       
        mu_dezinger: a dezinging parameter, when 0.0 - median filter, when > 0.0 - dezinger
        ncores: The number of CPU cores per process

    Returns:
        tuple[ndarray, ndarray, ndarray]: A tuple of numpy arrays containing the
            filtered projections, flatfields and darkfields.
    """
    from larix.methods.misc import MEDIAN_FILT, MEDIAN_DEZING

    if mu_dezinger > 0.0:
        data = MEDIAN_DEZING(data, radius_kernel, mu_dezinger*(np.std(data)), ncores)
    else:
        data = MEDIAN_FILT(data, radius_kernel, ncores)
    if flats is not None:
        if mu_dezinger > 0.0:
            flats = MEDIAN_DEZING(flats, radius_kernel, mu_dezinger*(np.std(data)), ncores)            
        else:
            flats = MEDIAN_FILT(flats, radius_kernel, ncores)            
    if darks is not None:
        if mu_dezinger > 0.0:
            darks = MEDIAN_DEZING(darks, radius_kernel, mu_dezinger*(np.std(data)), ncores)            
        else:            
            darks = MEDIAN_FILT(darks, radius_kernel, ncores)

    return data, flats, darks