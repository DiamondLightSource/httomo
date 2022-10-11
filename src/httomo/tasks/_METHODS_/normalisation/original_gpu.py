from cupy import isinf, isnan, log, mean, ndarray


def normalize_data(data: ndarray, darks: ndarray, flats: ndarray) -> ndarray:
    """Performs image normalization with reference to flatfields and darkfields.

    Returns:
        ndarray: A cupy array of normalized projections.
    """
    dark0 = mean(darks, axis=0)
    flat0 = mean(flats, axis=0)
    data = (data - dark0) / (flat0 - dark0 + 1e-3)
    data[data <= 0] = 1
    data = -log(data)
    data[isnan(data)] = 6.0
    data[isinf(data)] = 0

    return data
