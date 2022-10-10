import cupy
from larix.methods.misc_gpu import MEDIAN_FILT_GPU


def filter_data(
    data: cupy.ndarray, flats: cupy.ndarray, darks: cupy.ndarray
) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
    """Performs median filtration on the data using the larix library.

    Args:
        data: A cupy array containing the sample projections.
        flats: A cupy array containing the flatfield projections.
        darks: A cupy array containing the dark projections.

    Returns:
        tuple[ndarray, ndarray, ndarray]: A tuple of cupy arrays containing the
            filtered projections, flatfields and darkfields.
    """
    data = cupy.asnumpy(data)
    flats = cupy.asnumpy(flats)
    darks = cupy.asnumpy(darks)

    kernel_size = 3  # full size kernel 3 x 3 x 3
    data = MEDIAN_FILT_GPU(data, kernel_size)
    flats = MEDIAN_FILT_GPU(flats, kernel_size)
    darks = MEDIAN_FILT_GPU(darks, kernel_size)

    data = cupy.asarray(data)
    flats = cupy.asarray(flats)
    darks = cupy.asarray(darks)

    return data, flats, darks
