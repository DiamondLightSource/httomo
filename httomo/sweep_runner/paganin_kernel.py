import numpy as np
import math

BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


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
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _paganin_filter_factor(energy, dist, alpha, w2):
    return 1 / (1 + (dist * alpha * _wavelength(energy) * w2 / (4 * PI)))


def _reciprocal_coord(pixel_size: float, num_grid: int) -> np.ndarray:
    n = num_grid - 1
    rc = np.arange(-n, num_grid, 2, dtype=np.float32)
    rc *= 2 * math.pi / (n * pixel_size)
    return rc


def _reciprocal_grid(pixel_size, nx, ny):
    # Sampling in reciprocal space.
    indx = _reciprocal_coord(pixel_size, nx)
    indy = _reciprocal_coord(pixel_size, ny)
    np.square(indx, out=indx)
    np.square(indy, out=indy)
    return np.add.outer(indx, indy)
