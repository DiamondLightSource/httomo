from mpi4py import MPI
import numpy as np
from mpi4py.MPI import Comm

gpu_enabled = False
try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
        gpu_enabled = True  # CuPy is installed and GPU is available
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp
except ImportError:
    import numpy as xp


def min_max_mean_std(data: xp.ndarray, comm: Comm):
    """calculating global statistics of the given array
    Args:
        data: (xp.ndarray): a numpy or cupy data array.
        comm: The MPI communicator to use.
    Returns:
        tuple[(float, float, float, float)]: (min, max, mean, std_var)
    """
    if isinstance(data, np.ndarray):
        data_not_numpyarray = False
    else:
        data_not_numpyarray = True
    # if CuPy array is being passed we need to convert it to numpy for mpi allreduce
    converted = False
    if gpu_enabled and data_not_numpyarray:
        converted = True
        data = xp.asnumpy(data)

    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)

    comm.Barrier()
    data = data.flatten()
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
    if converted:
        # convert numpy array back to CuPy one
        data = xp.asarray(data)
    return (minval_glob, maxval_glob, cmean_glob, sigma_glob)
