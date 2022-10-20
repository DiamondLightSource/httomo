from numpy import ndarray
from tomopy import prep


def stripes(data: ndarray, method_name: str, ncores: int) -> ndarray:
    """Wrapper for tomopy.prep.stripes module.

    Args:
        data: A numpy array of projections.
        method_name: the name of method as in tomopy.prep.stripe
        ncores: The number of CPU cores per process

    Returns:
        ndarray: A numpy array of projections with stripes removed.
    """
    module = getattr(prep, 'stripes')
    data = getattr(module, method_name)(data, ncore=ncores)
    return data


def normalize(data: ndarray, flats: ndarray, darks: ndarray,
              ncores: int) -> ndarray:
    """Wrapper for tomopy.prep.normalize module.

    Args:
        data: A numpy array containing the sample projections.
        flats: A numpy array containing the flatfield projections.
        darks: A numpy array containing the dark projections.
        ncores: The number of CPU cores per process

    Returns:
        ndarray: A numpy array of normalized projections.
    """
    module = getattr(prep, 'normalize')
    data = getattr(module, 'normalize')(data, flats, dark, ncore=ncores,
                                        cutoff=10)
    data[data == 0.0] = 1e-09
    data = getattr(module, 'minus_log')(data, ncore=ncores)

    return data
