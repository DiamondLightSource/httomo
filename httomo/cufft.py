import ctypes
import sys
from enum import Enum


if "linux" not in sys.platform:
    raise RuntimeError("Linux is currently the only supported platform")

_linux_version_list = [
    "11.3.0.4",
    11.0,
    10.1,
    10.0,
    9.2,
    9.1,
    9.0,
    8.0,
    7.5,
    7.0,
    6.5,
    6.0,
    5.5,
    5.0,
    4.0,
]
_libcufft_libname_list = ["libcufft.so"] + [
    "libcufft.so.%s" % v for v in _linux_version_list
]

# Load library
_libcufft = None
for _libcufft_libname in _libcufft_libname_list:
    try:
        _libcufft = ctypes.cdll.LoadLibrary(_libcufft_libname)
    except OSError:
        pass
    else:
        break

# Print understandable error message when library cannot be found:
if _libcufft is None:
    raise OSError("cufft library not found")


# General CUFFT error
class CufftError(Exception):
    """CUFFT error"""

    pass


# Errors that can be returned by plan estimation functions
class CufftAllocFailed(CufftError):
    """CUFFT failed to allocate GPU memory."""

    pass


class CufftInvalidValue(CufftError):
    """The user specified a bad memory pointer."""

    pass


class CufftInternalError(CufftError):
    """Internal driver error."""

    pass


class CufftSetupFailed(CufftError):
    """The CUFFT library failed to initialize."""

    pass


class CufftInvalidSize(CufftError):
    """The user specified an unsupported FFT size."""

    pass


cufftExceptions = {
    0x2: CufftAllocFailed,
    0x4: CufftInvalidValue,
    0x5: CufftInternalError,
    0x7: CufftSetupFailed,
    0x8: CufftInvalidSize,
}


class _types:
    """Some alias types."""

    plan = ctypes.c_int
    stream = ctypes.c_void_p
    worksize = ctypes.c_size_t


# Data transformation types
class CufftType(Enum):
    CUFFT_R2C = 0x2A
    CUFFT_C2R = 0x2C
    CUFFT_C2C = 0x29
    CUFFT_D2Z = 0x6A
    CUFFT_Z2D = 0x6C
    CUFFT_Z2Z = 0x69


def cufftCheckStatus(status: int):
    """Raise an exception if the specified CUBLAS status is an error."""
    if status != 0:
        try:
            e = cufftExceptions[status]
        except KeyError:
            raise CufftError
        else:
            raise e


_libcufft.cufftEstimate1d.restype = int
_libcufft.cufftEstimate1d.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]


def cufft_estimate_1d(nx: int, fft_type: CufftType, batch: int = 1):
    """
    Return estimated work area for 1D FFT.

    References
    ----------
    `cufftEstimate1d <http://docs.nvidia.com/cuda/cufft/#function-cufftestimate1d>`_
    """
    worksize = _types.worksize()
    assert _libcufft is not None
    status = _libcufft.cufftEstimate1d(
        nx, fft_type.value, batch, ctypes.byref(worksize)
    )
    cufftCheckStatus(status)
    return worksize.value


_libcufft.cufftEstimate2d.restype = int
_libcufft.cufftEstimate2d.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]


def cufft_estimate_2d(nx: int, ny: int, fft_type: CufftType):
    """
    Return estimated work area for 2D FFT.

    References
    ----------
    `cufftEstimate2d <http://docs.nvidia.com/cuda/cufft/#function-cufftestimate2d>`_
    """
    worksize = _types.worksize()
    assert _libcufft is not None
    status = _libcufft.cufftEstimate2d(nx, ny, fft_type.value, ctypes.byref(worksize))
    cufftCheckStatus(status)
    return worksize.value
