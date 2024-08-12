from typing import Tuple
import numpy as np

__all__ = [
    "_calc_memory_bytes_rescale_to_int",
]


def _calc_memory_bytes_rescale_to_int(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    bits: int = kwargs["bits"]
    if bits == 8:
        itemsize = 1
    elif bits == 16:
        itemsize = 2
    else:
        itemsize = 4
    safety = 128
    return (
        int(np.prod(non_slice_dims_shape)) * (dtype.itemsize + itemsize) + safety,
        0,
    )
