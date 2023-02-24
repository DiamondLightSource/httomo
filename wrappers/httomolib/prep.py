from typing import Dict
from inspect import signature

from numpy import ndarray
import cupy as cp


from httomo.utils import pattern, Pattern
from httomolib import prep


@pattern(Pattern.projection)
def normalize(
    params: Dict, method_name: str, data: ndarray, flats: ndarray, darks: ndarray
) -> ndarray:
    """Wrapper for httomolib.prep.normalize module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped httomolib function that are
        independent of HTTomo.
    method_name : str
        The name of the method to use in httomolib.prep.normalize.
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
    module = getattr(prep, "normalize")

    # as now this function does not require ncore parameter
    # TODO: not elegant, needs rethinking
    try:
        del params["ncore"]
    except:
        pass

    cp._default_memory_pool.free_all_blocks()

    data = getattr(module, method_name)(
        cp.asarray(data), cp.asarray(flats), cp.asarray(darks), **params
    )
    return cp.asnumpy(data)


@pattern(Pattern.sinogram)
def stripe(params: Dict, method_name: str, data: ndarray) -> ndarray:
    """Wrapper for httomolib.prep.stripe module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped httomolib function that are
        independent of httomo.
    method_name : str
        The name of the method to use in  httomolib.prep.phase.
    data : ndarray
        A numpy array of projections.

    Returns
    -------
    ndarray
        A numpy array of projections with the stripes removed.
    """
    module = getattr(prep, "stripe")

    # as now this function does not require ncore parameter
    # TODO: not elegant, needs rethinking
    try:
        del params["ncore"]
    except:
        pass

    cp._default_memory_pool.free_all_blocks()

    data = getattr(module, method_name)(cp.asarray(data), **params)
    return cp.asnumpy(data)


@pattern(Pattern.projection)
def phase(params: Dict, method_name: str, data: ndarray) -> ndarray:
    """Wrapper for httomolib.prep.phase module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped httomolib function that are
        independent of httomo.
    method_name : str
        The name of the method to use in  httomolib.prep.phase.
    data : ndarray
        A numpy array of projections.

    Returns
    -------
    ndarray
        A numpy array of projections with phase-contrast enhancement.
    """
    module = getattr(prep, "phase")

    # as now this function does not require ncore parameter
    # TODO: not elegant, needs rethinking
    try:
        del params["ncore"]
    except:
        pass

    cp._default_memory_pool.free_all_blocks()

    data = getattr(module, method_name)(cp.asarray(data), **params)
    return cp.asnumpy(data)
