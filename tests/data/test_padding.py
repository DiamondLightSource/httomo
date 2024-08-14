from typing import List
import numpy as np
import pytest

from httomo.data.padding import extrapolate_after, extrapolate_before
from httomo.preview import PreviewConfig, PreviewDimConfig


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
def test_extrapolate_before_previewed(slicing_dim: int):
    GLOBAL_SHAPE = (180, 128, 160)
    UNPADDED_BLOCK_LENGTH = 5
    BEFORE_PADDING = 3
    # Note the cropping in the `detector_y` dimension that shifts the start of the data from
    # index 0 to index 10
    PREVIEW_CONFIG = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=180),
        detector_y=PreviewDimConfig(start=10, stop=128),
        detector_x=PreviewDimConfig(start=0, stop=160),
    )
    DIMS = [0, 1, 2]
    non_slicing_dims = tuple(set(DIMS) - set([slicing_dim]))
    global_data = np.arange(np.prod(GLOBAL_SHAPE), dtype=np.float32).reshape(
        GLOBAL_SHAPE
    )

    # Due to the cropping being in the `detector_y` dimension:
    # - if padding is in the `angles` dim, then the cropping affects the block shape's
    # non-slicing dims
    # - if padding is in the `detector_y` dim, then the cropping doesn't affect the block
    # shape's non-slicing dims
    padded_block_shape: List[int] = list(GLOBAL_SHAPE)
    padded_block_shape[slicing_dim] = UNPADDED_BLOCK_LENGTH + BEFORE_PADDING
    if slicing_dim == 0:
        for dim in non_slicing_dims:
            padded_block_shape[dim] = (
                PREVIEW_CONFIG[dim].stop - PREVIEW_CONFIG[dim].start
            )

    # Setup numpy array containing the expected values after modification by
    # `extrapolate_before()` (namely, that the "before" padded area has been filled in, and
    # nothing else in the `expected_padded_block` array has been changed)
    expected_padded_block = np.zeros(padded_block_shape, dtype=np.float32)

    # If the slicing/padding dim is the `angles` dim, then the slice to repeat for the block's
    # "before" padded area is the 0th slice in the `angles` dim, with `detector_y` cropped to
    # start at index 10.
    #
    # If the slicing/padding dim is the `detector_y` dim, because the cropping is also in the
    # `detector_y` dim, the padding needs to applied more carefully. The slice to be repeated
    # for the "before" padded area needs to be the start of the cropped region along the
    # `detector_y` dimension (ie, the slice with index 10).
    slices_read = [slice(None)] * 3
    slices_read[slicing_dim] = slice(
        PREVIEW_CONFIG[slicing_dim].start,
        PREVIEW_CONFIG[slicing_dim].start + 1,
    )

    if slicing_dim == 0:
        for dim in non_slicing_dims:
            slices_read[dim] = slice(
                PREVIEW_CONFIG[dim].start, PREVIEW_CONFIG[dim].stop
            )

    slices_write = [slice(None)] * 3
    slices_write[slicing_dim] = slice(BEFORE_PADDING)
    expected_padded_block[slices_write[0], slices_write[1], slices_write[2]] = (
        global_data[slices_read[0], slices_read[1], slices_read[2]]
    )

    # Block to have its "before" padded area filled in by the extrapolation function
    block_data = np.zeros(padded_block_shape, dtype=np.float32)
    extrapolate_before(
        global_data,
        block_data,
        BEFORE_PADDING,
        slicing_dim,
        preview_config=PREVIEW_CONFIG,
    )
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


@pytest.mark.parametrize(
    "slicing_dim",
    [0, 1],
    ids=["projection-padding", "sinogram-padding"],
)
def test_extrapolate_after_previewed(slicing_dim: int):
    GLOBAL_SHAPE = (180, 128, 160)
    UNPADDED_BLOCK_LENGTH = 5
    AFTER_PADDING = 3
    # Note the cropping in the `detector_y` dimension that shifts the end of the data from
    # index 128 to index 118
    PREVIEW_CONFIG = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=180),
        detector_y=PreviewDimConfig(start=0, stop=118),
        detector_x=PreviewDimConfig(start=0, stop=160),
    )
    DIMS = [0, 1, 2]
    non_slicing_dims = tuple(set(DIMS) - set([slicing_dim]))
    global_data = np.arange(np.prod(GLOBAL_SHAPE), dtype=np.float32).reshape(
        GLOBAL_SHAPE
    )

    # Due to the cropping being in the `detector_y` dimension:
    # - if padding is in the `angles` dim, then the cropping affects the block shape's
    # non-slicing dims
    # - if padding is in the `detector_y` dim, then the cropping doesn't affect the block
    # shape's non-slicing dims
    padded_block_shape: List[int] = list(GLOBAL_SHAPE)
    padded_block_shape[slicing_dim] = UNPADDED_BLOCK_LENGTH + AFTER_PADDING
    if slicing_dim == 0:
        for dim in non_slicing_dims:
            padded_block_shape[dim] = (
                PREVIEW_CONFIG[dim].stop - PREVIEW_CONFIG[dim].start
            )

    # Setup numpy array containing the expected values after modification by
    # `extrapolate_after()` (namely, that the "after" padded area has been filled in, and
    # nothing else in the `expected_padded_block` array has been changed)
    expected_padded_block = np.zeros(padded_block_shape, dtype=np.float32)

    # If the slicing/padding dim is the `angles` dim, then the the slice to repeat for the
    # block's "after" padded area is the last slice in the `angles` dim, with `detector_y`
    # cropped to end at index 117.
    #
    # If the slicing/padding dim is the `detector_y` dim, because the cropping is also in the
    # `detector_y` dim, the padding needs to be applied more carefully. The slice to be
    # repeated for the "after" padded area needs to be the end of the cropped region along the
    # `detector_y` dim (ie, the slice with index 117).
    slices_read = [slice(None)] * 3
    slices_read[slicing_dim] = slice(
        PREVIEW_CONFIG[slicing_dim].stop - 1,
        PREVIEW_CONFIG[slicing_dim].stop,
    )

    if slicing_dim == 0:
        for dim in non_slicing_dims:
            slices_read[dim] = slice(
                PREVIEW_CONFIG[dim].start, PREVIEW_CONFIG[dim].stop
            )

    slices_write = [slice(None)] * 3
    slices_write[slicing_dim] = slice(
        padded_block_shape[slicing_dim] - AFTER_PADDING, padded_block_shape[slicing_dim]
    )
    expected_padded_block[slices_write[0], slices_write[1], slices_write[2]] = (
        global_data[slices_read[0], slices_read[1], slices_read[2]]
    )

    # Block to have its "after" padded area filled in by the extrapolation function
    block_data = np.zeros(padded_block_shape, dtype=np.float32)
    extrapolate_after(
        global_data,
        block_data,
        AFTER_PADDING,
        slicing_dim,
        preview_config=PREVIEW_CONFIG,
    )
    np.testing.assert_array_equal(block_data, expected_padded_block)
