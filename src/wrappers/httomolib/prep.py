from typing import Dict
from inspect import signature

from numpy import ndarray
# a workaround to expose functions in prep. Needs investigation why you cannot just do: from httomolib import prep as with tomopy
from httomolib.prep.phase import *
from httomolib import prep

def phase(params: Dict, method_name: str, data: ndarray) -> ndarray:
    """Wrapper for httomolib.prep.phase module.

    Parameters
    ----------
    params : Dict
        A dict containing all params of the wrapped tomopy function that are
        independent of httomo.
    method_name : str
        The name of the method to use in  httomolib.prep.phase.
    data : ndarray
        A numpy array of projections.

    Returns
    -------
    ndarray
        A numpy array of projections with stripes removed.
    """
    module = getattr(prep, 'phase')
    data = getattr(module, method_name)(data, **params)
    return data