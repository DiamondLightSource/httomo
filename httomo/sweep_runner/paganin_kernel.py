import numpy as np
import math
from scipy.signal import peak_widths, find_peaks
from scipy.fft import fftshift

BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e2  # [cm/s]
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def paganin_kernel_estimator(
    pixel_size: float,
    distance: float,    
    energy: float,    
    ratio_delta_beta: tuple,
    vert_min_limit: int,
    peak_height: float,
) -> np.ndarray:
    """
    Using functions from Paganin filter to estimate the width of the kernel
    """
    extended_dim_size = 4096
    padded_shape_dy = extended_dim_size  # we assume here the size of the padded projection to be pow(2,12)
    padded_shape_dx = extended_dim_size

    # Compute the reciprocal grid
    indx = _reciprocal_coord(pixel_size, padded_shape_dy)
    indy = _reciprocal_coord(pixel_size, padded_shape_dx)

    kernels = np.zeros(len(ratio_delta_beta), dtype=int)
    for count, ratio_delta_beta_scalar in enumerate(ratio_delta_beta):

        # calculate alpha constant
        alpha = _calculate_alpha(energy, distance / 1e-6, ratio_delta_beta_scalar)

        phase_filter = fftshift(
        1.0 / (1.0 + alpha * (np.add.outer(np.square(indx), np.square(indy))))
        )

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


def _calculate_alpha(energy, distance_micron, ratio_delta_beta):
    return (
        _wavelength_micron(energy) * distance_micron / (4 * math.pi)
    ) * ratio_delta_beta



def _wavelength_micron(energy: float) -> float:
    SPEED_OF_LIGHT = 299792458e2 * 10000.0  # [microns/s]
    PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
    return 2 * math.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _reciprocal_coord(pixel_size: float, num_grid: int) -> np.ndarray:
    n = num_grid - 1
    rc = np.arange(-n, num_grid, 2, dtype=np.float32)
    rc *= 2 * math.pi / (n * pixel_size)
    return rc