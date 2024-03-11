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

    # fixed cost for input/output of darks/flats - it makes a copy of these
    # As we process data/flats/darks sequentially, the space is normally freed after 
    # one iteration so there is no need to account for d/f in this function. 
    # We also assume here as with data_reducer that you need to have a GPU card with the memory 
    # enough to to fit either darks or flats.     
    
    # it does not change the data type of the inputs
    in_slice_mem = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_slice_mem = in_slice_mem   
   
    tot_memory_bytes = int(in_slice_mem + out_slice_mem)

    return (tot_memory_bytes, 0)