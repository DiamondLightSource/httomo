from numpy import ndarray
from tomopy import minus_log, normalize


def normalise_tomopy(data: ndarray, flats: ndarray, darks: ndarray, ncores: int) -> ndarray:
    """Performs image normalization with reference to flatfields and darkfields.

    Args:
        data: A numpy array containing the sample projections.
        flats: A numpy array containing the flatfield projections.
        darks: A numpy array containing the dark projections.
        ncores: The number of CPU cores per process

    Returns:
        ndarray: A numpy array of normalized projections.
    """
    data = normalize(data, flats, darks, ncore=ncores, cutoff=10)
    data[data == 0.0] = 1e-09
    data = minus_log(data, ncore=ncores)

    return data
