from numpy import ndarray
from tomopy.prep.stripe import remove_stripe_ti


def remove_stripes(data: ndarray) -> ndarray:
    """Removes stripes with tomopy's remove_stripe_ti.

    Args:
        data: A numpy array of projections.

    Returns:
        ndarray: A numpy array of projections with stripes removed.
    """
    return remove_stripe_ti(data, nblock=0, alpha=1.5, ncore=1)
