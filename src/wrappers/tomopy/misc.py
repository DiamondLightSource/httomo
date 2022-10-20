from numpy import ndarray
from tomopy import misc


def corr(data: ndarray, method_name: str, ncores: int) -> ndarray:
    """Wrapper for tomopy.misc.corr module.

    Args:
        data: A numpy array of projections.
        method_name: the name of method as in tomopy.misc.corr
        ncores: The number of CPU cores per process

    Returns:
        ndarray: A numpy array of projections with the correction method
                 applied.
    """
    module = getattr(prep, 'corr')
    data = getattr(module, method_name)(data, ncore=ncores)
    return data
