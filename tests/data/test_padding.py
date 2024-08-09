from typing import List
import numpy as np
import pytest

from httomo.data.padding import extrapolate_after, extrapolate_before


@pytest.mark.parametrize(
    "slicing_dim",
    [0, 1],
    ids=["projection-padding", "sinogram-padding"],
)
def test_extrapolate_before(slicing_dim: int):
    GLOBAL_SHAPE = (180, 128, 160)
    UNPADDED_BLOCK_LENGTH = 5
    BEFORE_PADDING = 3

    # Setup the block data to be padded with values from the parent chunk in the global data
    padded_block_shape: List[int] = list(GLOBAL_SHAPE)
    padded_block_shape[slicing_dim] = UNPADDED_BLOCK_LENGTH + BEFORE_PADDING
    global_data = np.arange(np.prod(GLOBAL_SHAPE), dtype=np.float32).reshape(
        GLOBAL_SHAPE
    )
    block_data = np.zeros(padded_block_shape, dtype=np.float32)

    # Setup numpy array containing the expected values after modification by
    # `extrapolate_before()` (namely, that the "before" padded area has been filled in, and
    # nothing else in the `expected_padded_block` array has been changed)
    expected_padded_block = np.zeros(padded_block_shape, dtype=np.float32)
    slices_read = [slice(None)] * 3
    slices_read[slicing_dim] = slice(1)
    slices_write = [slice(None)] * 3
    slices_write[slicing_dim] = slice(BEFORE_PADDING)
    expected_padded_block[slices_write[0], slices_write[1], slices_write[2]] = (
        global_data[slices_read[0], slices_read[1], slices_read[2]]
    )

    extrapolate_before(global_data, block_data, BEFORE_PADDING, slicing_dim)
    np.testing.assert_array_equal(block_data, expected_padded_block)


@pytest.mark.parametrize(
    "slicing_dim",
    [0, 1],
    ids=["projection-padding", "sinogram-padding"],
)
def test_extrapolate_after(slicing_dim: int):
    GLOBAL_SHAPE = (180, 128, 160)
    UNPADDED_BLOCK_LENGTH = 5
    AFTER_PADDING = 3

    # Setup the block data to be padded with values from the parent chunk in the global data
    padded_block_shape: List[int] = list(GLOBAL_SHAPE)
    padded_block_shape[slicing_dim] = UNPADDED_BLOCK_LENGTH + AFTER_PADDING
    global_data = np.arange(np.prod(GLOBAL_SHAPE), dtype=np.float32).reshape(
        GLOBAL_SHAPE
    )
    block_data = np.zeros(padded_block_shape, dtype=np.float32)

    # Setup numpy array containing the expected values after modification by
    # `extrapolate_after()` (namely, that the "after" padded area has been filled in, and
    # nothing else in the `expected_padded_block` array has been changed)
    expected_padded_block = np.zeros(padded_block_shape, dtype=np.float32)
    slices_read = [slice(None)] * 3
    slices_read[slicing_dim] = slice(
        GLOBAL_SHAPE[slicing_dim] - 1, GLOBAL_SHAPE[slicing_dim]
    )
    slices_write = [slice(None)] * 3
    slices_write[slicing_dim] = slice(
        padded_block_shape[slicing_dim] - AFTER_PADDING, padded_block_shape[slicing_dim]
    )
    expected_padded_block[slices_write[0], slices_write[1], slices_write[2]] = (
        global_data[slices_read[0], slices_read[1], slices_read[2]]
    )

    extrapolate_after(global_data, block_data, AFTER_PADDING, slicing_dim)
    np.testing.assert_array_equal(block_data, expected_padded_block)
