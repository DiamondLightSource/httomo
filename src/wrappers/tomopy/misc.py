from typing import Dict

from numpy import ndarray
from tomopy import misc


def corr(method_name: str, data: ndarray, params: Dict) -> ndarray:
    """Wrapper for tomopy.misc.corr module.

    Args:
        method_name: The name of the method to use in tomopy.misc.corr
        data: A numpy array of projections.
        params: A dict containing all params of the wrapped tomopy function that
                are not related to the data loaded by a loader function

    Returns:
        ndarray: A numpy array of projections with the correction method
                 applied.
    """
    module = getattr(misc, 'corr')
    data = getattr(module, method_name)(data, **params)
    return data
