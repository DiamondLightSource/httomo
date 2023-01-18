from typing import Dict
from inspect import signature

from numpy import ndarray
import cupy as cp

from httomolib import recon

def algorithm(params: Dict,
              method_name: str, 
              data: ndarray,
              angles_radians: ndarray,
              gpu_id: int) -> ndarray:
    """Wrapper for httomolib.recon.algorithm module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped httomolib function that are
        independent of httomo.
    method_name : str
        The name of the method to use in  httomolib.recon.algorithm.
    data : ndarray
        A numpy array of projections.
    angles_radians : ndarray
        A numpy array of angles in radians.
    gpu_id : int
        A GPU device index to execute operation on.        

    Returns
    -------
    ndarray
        A numpy array of projections with stripes removed.
    """
    module = getattr(recon, 'algorithm')
    data = getattr(module, method_name)(data, angles=angles_radians, gpu_id=gpu_id, **params)
    return data