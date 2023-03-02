from typing import Dict

from numpy import ndarray
from tomopy import misc


def corr(params: Dict, method_name: str, data: ndarray) -> ndarray:
    """Wrapper for tomopy.misc.corr module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped tomopy function that are
        independent of HTTomo.
    method_name : str
        The name of the method to use in tomopy.misc.corr.
    data : ndarray
        A numpy array of projections.

    Returns
    -------
    ndarray
        A numpy array of projections with the correction method applied.
    """
    module = getattr(misc, 'corr')
    data = getattr(module, method_name)(data, **params)
    return data


def morph(params: Dict, method_name: str, data: ndarray) -> ndarray:
    """Wrapper for tomopy.misc.morph module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped tomopy function that are
        independent of HTTomo.
    method_name : str
        The name of the method to use in tomopy.misc.morph.
    data : ndarray
        A numpy array of projections.

    Returns
    -------
    ndarray
        A numpy array of projections with the correction method applied.
    """

    # as now this function does not require ncore parameter 
    # TODO: not elegant, needs rethinking
    try:
        del params["ncore"]
    except:
        pass
 
    module = getattr(misc, 'morph')
    data = getattr(module, method_name)(data, **params)
    return data