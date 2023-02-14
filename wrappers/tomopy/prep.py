from typing import Dict
from inspect import signature

from numpy import ndarray
from tomopy import prep

from httomo.utils import Pattern, pattern


@pattern(Pattern.sinogram)
def stripe(params: Dict, method_name: str, data: ndarray) -> ndarray:
    """Wrapper for tomopy.prep.stripe module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped tomopy function that are
        independent of HTTomo.
    method_name : str
        The name of the method to use in tomopy.prep.stripe.
    data : ndarray
        A numpy array of projections.

    Returns
    -------
    ndarray
        A numpy array of projections with stripes removed.
    """
    module = getattr(prep, 'stripe')
    data = getattr(module, method_name)(data, **params)
    return data


@pattern(Pattern.projection)
def normalize(params: Dict, method_name: str, data: ndarray, flats: ndarray,
              darks: ndarray) -> ndarray:
    """Wrapper for tomopy.prep.normalize module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped tomopy function that are
        independent of HTTomo.
    method_name : str
        The name of the method to use in tomopy.prep.normalize.
    data : ndarray
        A numpy array containing the sample projections.
    flats :ndarray
        A numpy array containing the flatfield projections.
    darks : ndarray
        A numpy array containing the dark projections.

    Returns
    -------
    ndarray
        A numpy array of normalized projections.
    """
    module = getattr(prep, 'normalize')
    # Some functions in `tomopy.prep.normalize` need the flats and darks, but
    # others do not, so this needs to be checked prior to passing the parameters
    # to the wrapped tomopy function
    func = getattr(module, method_name)
    sig_params = signature(func).parameters
    if 'dark' in sig_params and 'flat' in sig_params:
        data = getattr(module, method_name)(data, flats, darks, **params)
    else:
        data = getattr(module, method_name)(data, **params)
    return data
