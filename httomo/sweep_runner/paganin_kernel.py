import numpy as np
from scipy.signal import peak_widths, find_peaks

BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e2  # [cm/s]
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def paganin_kernel_estimator(
    pixel_size: float,
    alpha_tuple: tuple,
    energy: float,
    dist: float,
    vert_min_limit: int,
    peak_height: float,
) -> np.ndarray:
    """
    Using functions from Paganin filter to estimate the width of the kernel
    """
    extended_dim_size = 4096
    padded_shape_dy = extended_dim_size  # we assume here the size of the padded projection to be pow(2,12)
    padded_shape_dx = extended_dim_size
    w2 = _reciprocal_grid(pixel_size, padded_shape_dy, padded_shape_dx)

    kernels = np.zeros(len(alpha_tuple), dtype=int)
    for count, alpha_scalar in enumerate(alpha_tuple):
        phase_filter = _paganin_filter_factor(energy, dist, alpha_scalar, w2)

        curve1D = np.abs(phase_filter[extended_dim_size // 2, :])
        peaks, _ = find_peaks(curve1D)
        full_size_kernel = int(
            peak_widths(curve1D, peaks=peaks, rel_height=peak_height)[0]
        )
        if full_size_kernel == 0:
            kernels[count] = vert_min_limit
        else:
            kernels[count] = full_size_kernel

    return kernels


def _calculate_pad_size(datashape: tuple) -> list:
    pad_list = []
    for index, element in enumerate(datashape):
        if index == 0:
            pad_width = (0, 0)  # do not pad the slicing dim
        else:
            diff = _shift_bit_length(element + 1) - element
            if element % 2 == 0:
                pad_width_scalar = diff // 2
                pad_width = (pad_width_scalar, pad_width_scalar)
            else:
                # need an uneven padding for odd-number lengths
                left_pad = diff // 2
                right_pad = diff - left_pad
                pad_width = (left_pad, right_pad)

        pad_list.append(pad_width)

    return pad_list


def _shift_bit_length(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _wavelength(energy):
    return 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _paganin_filter_factor(energy, dist, alpha, w2):
    return 1 / (1 + (dist * alpha * _wavelength(energy) * w2 / (4 * np.pi)))


def _reciprocal_coord(pixel_size: float, num_grid: int) -> np.ndarray:
    n = num_grid - 1
    rc = np.arange(-n, num_grid, 2, dtype=np.float32)
    rc *= 2 * np.pi / (n * pixel_size)
    return rc


def _reciprocal_grid(pixel_size, nx, ny):
    # Sampling in reciprocal space.
    indx = _reciprocal_coord(pixel_size, nx)
    indy = _reciprocal_coord(pixel_size, ny)
    np.square(indx, out=indx)
    np.square(indy, out=indy)
    return np.add.outer(indx, indy)
