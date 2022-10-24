from typing import Dict

from numpy import ndarray
from tomopy import prep


def stripe(params: Dict, method_name: str, data: ndarray) -> ndarray:
    """Wrapper for tomopy.prep.stripe module.

    Args:
        params: A dict containing all params of the wrapped tomopy function that
                are not related to the data loaded by a loader function
        method_name: The name of the method to use in tomopy.prep.stripe
        data: A numpy array of projections.

    Returns:
        ndarray: A numpy array of projections with stripes removed.
    """
    module = getattr(prep, 'stripe')
    data = getattr(module, method_name)(data, **params)
    return data


def normalize(params: Dict, method_name: str, data: ndarray, flats: ndarray,
              darks: ndarray) -> ndarray:
    """Wrapper for tomopy.prep.normalize module.

    Args:
        params: A dict containing all params of the wrapped tomopy function that
                are not related to the data loaded by a loader function
        method_name: The name of the method to use in tomopy.prep.normalize
        data: A numpy array containing the sample projections.
        flats: A numpy array containing the flatfield projections.
        darks: A numpy array containing the dark projections.

    Returns:
        ndarray: A numpy array of normalized projections.
    """
    module = getattr(prep, 'normalize')
    data = getattr(module, method_name)(data, flats, darks, **params)
    data[data == 0.0] = 1e-09
    return data
