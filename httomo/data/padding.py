import h5py
import numpy as np


def extrapolate_before(
    global_data: h5py.Dataset | np.ndarray,
    block_data: np.ndarray,
    slices: int,
    dim: int,
    offset: int = 0,
) -> None:
    """
    Read the required "before" padded area for a block into the given numpy array that
    represents the block.

    NOTE: Currently performs "edge" padding as is understood in the padding terminology of
    `np.pad()`
    """
    if slices == 0:
        return
    slices_wrt = [slice(None), slice(None), slice(None)]
    slices_wrt[dim] = slice(slices)
    slices_read = [slice(None), slice(None), slice(None)]
    slices_read[dim] = slice(offset, offset + 1)
    block_data[slices_wrt[0], slices_wrt[1], slices_wrt[2]] = global_data[
        slices_read[0], slices_read[1], slices_read[2]
    ]


def extrapolate_after(
    global_data: h5py.Dataset | np.ndarray,
    block_data: np.ndarray,
    slices: int,
    dim: int,
    offset: int = 0,
):
    """
    Read the required "after" padded area for a block into the given numpy array that
    represents the block.

    NOTE: Currently performs "edge" padding as is understood in the padding terminology of
    `np.pad()`
    """
    if slices == 0:
        return
    slices_wrt = [slice(None), slice(None), slice(None)]
    slices_wrt[dim] = slice(block_data.shape[dim] - slices, block_data.shape[dim])
    slices_read = [slice(None), slice(None), slice(None)]
    slices_read[dim] = slice(
        global_data.shape[dim] - 1 - offset, global_data.shape[dim] - offset
    )
    block_data[slices_wrt[0], slices_wrt[1], slices_wrt[2]] = global_data[
        slices_read[0], slices_read[1], slices_read[2]
    ]
