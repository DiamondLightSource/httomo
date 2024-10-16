from typing import Optional, Union

import h5py
import numpy as np

from httomo.preview import PreviewConfig


DIMS = [0, 1, 2]


def extrapolate_before(
    global_data: Union[h5py.Dataset, np.ndarray],
    block_data: np.ndarray,
    slices: int,
    dim: int,
    offset: int = 0,
    preview_config: Optional[PreviewConfig] = None,
) -> None:
    """
    Read the required "before" padded area for a block into the given numpy array that
    represents the block.

    NOTE: Currently performs "edge" padding as is understood in the padding terminology of
    `np.pad()`
    """
    if slices == 0:
        return

    non_slicing_dims = tuple(set(DIMS) - set([dim]))
    slices_wrt = [slice(None), slice(None), slice(None)]
    slices_wrt[dim] = slice(slices)

    slices_read = [slice(None), slice(None), slice(None)]
    if preview_config is None:
        slices_read[dim] = slice(offset, offset + 1)
    else:
        slices_read[dim] = slice(
            preview_config[dim].start, preview_config[dim].start + 1
        )
        for non_slicing_dim in non_slicing_dims:
            slices_read[non_slicing_dim] = slice(
                preview_config[non_slicing_dim].start,
                preview_config[non_slicing_dim].stop,
            )

    block_data[slices_wrt[0], slices_wrt[1], slices_wrt[2]] = global_data[
        slices_read[0], slices_read[1], slices_read[2]
    ]


def extrapolate_after(
    global_data: Union[h5py.Dataset, np.ndarray],
    block_data: np.ndarray,
    slices: int,
    dim: int,
    offset: int = 0,
    preview_config: Optional[PreviewConfig] = None,
):
    """
    Read the required "after" padded area for a block into the given numpy array that
    represents the block.

    NOTE: Currently performs "edge" padding as is understood in the padding terminology of
    `np.pad()`
    """
    if slices == 0:
        return

    non_slicing_dims = tuple(set(DIMS) - set([dim]))
    slices_wrt = [slice(None), slice(None), slice(None)]
    slices_wrt[dim] = slice(block_data.shape[dim] - slices, block_data.shape[dim])

    slices_read = [slice(None), slice(None), slice(None)]
    if preview_config is None:
        slices_read[dim] = slice(
            global_data.shape[dim] - 1 - offset, global_data.shape[dim] - offset
        )
    else:
        slices_read[dim] = slice(preview_config[dim].stop - 1, preview_config[dim].stop)
        for non_slicing_dim in non_slicing_dims:
            slices_read[non_slicing_dim] = slice(
                preview_config[non_slicing_dim].start,
                preview_config[non_slicing_dim].stop,
            )

    block_data[slices_wrt[0], slices_wrt[1], slices_wrt[2]] = global_data[
        slices_read[0], slices_read[1], slices_read[2]
    ]
