from typing import Tuple
import numpy as np


__all__ = [
    "_calc_memory_bytes_remove_outlier3d",
]

def _calc_memory_bytes_remove_outlier3d(
        non_slice_dims_shape: Tuple[int, int],
        dtype: np.dtype,
        **kwargs,
) -> Tuple[int, int]:
    # this function is implicitly replaced with the dezinging wrapper, 
    # which applies it 3 times - to data, darks, and flats
    
    # it does not change the data type of the inputs
    in_slice_mem = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_slice_mem = in_slice_mem
    
    # fixed cost for input/output of darks/flats - it makes a copy of these
    darks_flats_mem = 0
    darks_shape = kwargs["darks_shape"]
    darks_dtype = kwargs["darks_dtype"]
    darks_mem = int(np.prod(darks_shape) * darks_dtype.itemsize)
    flats_mem = int(np.prod(darks_shape) * darks_dtype.itemsize)
    
    # for memory calculation, we have to take into account that the input data is dropped after
    # an execution, freeing the memory for the next call afterwards. 
    # so if darks/flats are smaller than data, they will re-use that memory that has just been freed
    #
    # Because we don't know the size of the data at this point, we can't be accurate here. For a 
    # conservative estimate, we assume that flats/darks memory comes in addition to the data memory.
    # But we only need to consider the greater of the two (the other will re-use)
    darks_flats_mem = max(darks_mem, flats_mem)

    tot_memory_bytes = int(in_slice_mem + out_slice_mem)

    return (tot_memory_bytes, darks_flats_mem)