from typing import Dict
import numpy as np
import cupy as cp
from mpi4py.MPI import Comm

from httomo.utils import pattern, Pattern
from httomolib import misc


@pattern(Pattern.all)
def images(params: Dict, method_name: str, out_dir: str, comm: Comm, data: np.ndarray) -> np.ndarray:
    """Wrapper for httomolib.misc.images module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped tomopy function that are
        independent of httomo.
    method_name : str
        The name of the method to use in httomolib.misc.corr.
    out_dir : str
        The output directory.
    comm: int
        the MPI communicator.
    data : ndarray
        A numpy data array.

    Returns
    -------
    """
    # as now this function does not require ncore parameter 
    del params["ncore"]
    
    module = getattr(misc, 'images')
    data = getattr(module, method_name)(data, out_dir, comm_rank = comm.rank, **params)
    return data


@pattern(Pattern.all)
def corr(params: Dict, method_name: str, data: np.ndarray, gpu_id: int) -> np.ndarray:
    """Wrapper for httomolib.misc.corr module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped httomolib function that are
        independent of httomo.
    method_name : str
        The name of the method to use in  httomolib.prep.phase.
    data : ndarray
        A numpy array of projections.
    gpu_id : int
        A GPU device index to execute operation on.        

    Returns
    -------
    ndarray
        A numpy array of corrected data.
    """
    module = getattr(misc, 'corr')

    # as now this function does not require ncore parameter 
    del params["ncore"]

    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device(gpu_id).use()
    
    data = getattr(module, method_name)(cp.asarray(data), **params)
    return cp.asnumpy(data)