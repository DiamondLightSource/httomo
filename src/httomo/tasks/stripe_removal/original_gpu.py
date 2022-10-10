import cupy


def remove_stripes(data: cupy.ndarray) -> cupy.ndarray:
    """Removes stripes with the method of V. Titarenko (TomoCuPy).

    Args:
        data: A cupy array of projections.

    Returns:
        cupy.ndarray: A cupy array of projections with stripes removed.
    """
    beta = 0.1  # lowering the value increases the filter strength
    gamma = beta * ((1 - beta) / (1 + beta)) ** cupy.abs(
        cupy.fft.fftfreq(data.shape[-1]) * data.shape[-1]
    )
    gamma[0] -= 1
    v = cupy.mean(data, axis=0)
    v = v - v[:, 0:1]
    v = cupy.fft.irfft(cupy.fft.rfft(v) * cupy.fft.rfft(gamma))
    data[:] += v

    return data
