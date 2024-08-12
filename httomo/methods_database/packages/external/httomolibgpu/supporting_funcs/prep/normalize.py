from typing import Tuple
import numpy as np


__all__ = [
    "_calc_memory_bytes_normalize",
]


def _calc_memory_bytes_normalize(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    # this function changes the data type
    in_slice_mem = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_slice_mem = np.prod(non_slice_dims_shape) * np.float32().itemsize

    # fixed cost for keeping mean of flats and darks
    mean_mem = int(np.prod(non_slice_dims_shape) * np.float32().itemsize * 2)

    tot_memory_bytes = int(in_slice_mem + out_slice_mem)

    return (tot_memory_bytes, mean_mem)
