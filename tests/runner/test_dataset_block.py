import math
from typing import Literal, Tuple

import pytest
import numpy as np

from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.utils import make_3d_shape_from_shape


def test_full_block_for_global_data():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(data=data, aux_data=AuxiliaryData(angles=angles))

    assert block.is_cpu is True
    assert block.is_gpu is False
    assert block.global_index == (0, 0, 0)
    assert block.global_shape == (10, 10, 10)
    assert block.chunk_index == (0, 0, 0)
    assert block.chunk_shape == (10, 10, 10)
    assert block.shape == (10, 10, 10)
    assert block.is_last_in_chunk is True
    assert block.slicing_dim == 0

    np.testing.assert_array_equal(data, block.data)
    np.testing.assert_array_equal(angles, block.angles)
    np.testing.assert_array_equal(angles, block.angles_radians)
    assert block.darks.shape == (0, 10, 10)
    assert block.dark.shape == (0, 10, 10)
    assert block.flats.shape == (0, 10, 10)
    assert block.flat.shape == (0, 10, 10)
    assert block.darks.dtype == data.dtype
    assert block.flats.dtype == data.dtype
    assert block.dark.dtype == data.dtype
    assert block.flat.dtype == data.dtype


def test_full_block_for_global_data_with_padding():
    padding = (2, 2)
    data = np.ones((14, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(data=data, aux_data=AuxiliaryData(angles=angles), padding=(2, 2), slicing_dim=0,
                         block_start=-2, chunk_start=-2)

    assert block.is_cpu is True
    assert block.is_gpu is False
    assert block.global_index == (-2, 0, 0)
    assert block.global_shape == (10, 10, 10)
    assert block.chunk_index == (-2, 0, 0)
    assert block.chunk_shape == (14, 10, 10)
    assert block.shape == (14, 10, 10)
    assert block.is_last_in_chunk is True
    assert block.slicing_dim == 0
    assert block.is_padded is True
    assert block.padding == padding

    np.testing.assert_array_equal(data, block.data)


@pytest.mark.parametrize("slicing_dim", [0, 1, 2])
def test_full_block_for_chunked_data(slicing_dim: Literal[0, 1, 2]):
    data = np.ones((10, 10, 10), dtype=np.float32)
    chunk_start = 10
    global_shape_t = [10, 10, 10]
    global_shape_t[slicing_dim] = 30
    global_shape = make_3d_shape_from_shape(global_shape_t)
    angles = np.linspace(0, math.pi, global_shape[0], dtype=np.float32)
    block = DataSetBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
        block_start=0,
        chunk_start=chunk_start,
        slicing_dim=slicing_dim,
        global_shape=global_shape,
    )

    expected_global_index = [0, 0, 0]
    expected_global_index[slicing_dim] = chunk_start
    assert block.global_index == tuple(expected_global_index)
    assert block.global_shape == global_shape
    assert block.chunk_index == (0, 0, 0)
    assert block.chunk_shape == (
        10,
        10,
        10,
    )  # assume that block spans whole chunk if not given
    assert block.shape == (10, 10, 10)
    assert block.is_last_in_chunk is True
    assert block.slicing_dim == slicing_dim


@pytest.mark.parametrize("slicing_dim", [0, 1, 2])
@pytest.mark.parametrize("last_in_chunk", [False, True], ids=["middle", "last"])
def test_partial_block_for_chunked_data(
    slicing_dim: Literal[0, 1, 2], last_in_chunk: bool
):
    block_shape = [10, 10, 10]
    block_shape[slicing_dim] = 2
    start_index = 3 if not last_in_chunk else 8
    data = np.ones(block_shape, dtype=np.float32)
    global_index = [0, 0, 0]
    global_index[slicing_dim] = 10 + start_index
    global_shape_t = [10, 10, 10]
    global_shape_t[slicing_dim] = 30
    global_shape = make_3d_shape_from_shape(global_shape_t)
    angles = np.linspace(0, math.pi, global_shape[0], dtype=np.float32)
    block = DataSetBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
        slicing_dim=slicing_dim,
        block_start=start_index,
        chunk_start=10,
        global_shape=global_shape,
        chunk_shape=(10, 10, 10),
    )

    assert block.global_index == tuple(global_index)
    assert block.global_shape == global_shape
    expected_chunk_index = [0, 0, 0]
    expected_chunk_index[slicing_dim] = start_index
    assert block.chunk_index == tuple(expected_chunk_index)
    assert block.chunk_shape == (10, 10, 10)
    assert block.shape == tuple(block_shape)
    assert block.is_last_in_chunk is last_in_chunk
    assert block.slicing_dim == slicing_dim


@pytest.mark.parametrize("slicing_dim", [0, 1, 2])
def test_partial_block_for_chunked_data_with_padding_center(
    slicing_dim: Literal[0, 1, 2]
):
    block_shape = [10, 10, 10]
    block_shape[slicing_dim] = 6
    start_index = 3
    data = np.ones(block_shape, dtype=np.float32)
    global_index = [0, 0, 0]
    global_index[slicing_dim] = 10 + start_index
    global_shape_t = [10, 10, 10]
    global_shape_t[slicing_dim] = 30
    global_shape = make_3d_shape_from_shape(global_shape_t)
    chunk_shape_t = [10, 10, 10]
    chunk_shape_t[slicing_dim] += 4  # for padding
    chunk_shape = make_3d_shape_from_shape(chunk_shape_t)
    angles = np.linspace(0, math.pi, global_shape[0], dtype=np.float32)
    block = DataSetBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
        slicing_dim=slicing_dim,
        block_start=start_index,
        chunk_start=10,
        global_shape=global_shape,
        chunk_shape=chunk_shape,
        padding=(2, 2),
    )

    assert block.is_padded is True
    assert block.padding == (2, 2)


@pytest.mark.parametrize("slicing_dim", [0, 1, 2])
@pytest.mark.parametrize("boundary", ["before", "after"])
def test_partial_block_for_chunked_data_with_padding_chunk_boundaries(
    slicing_dim: Literal[0, 1, 2], boundary: Literal["before", "after"]
):
    block_shape = [10, 10, 10]
    block_shape[slicing_dim] = 6
    start_index = -2 if boundary == "before" else 6
    data = np.ones(block_shape, dtype=np.float32)
    global_index = [0, 0, 0]
    global_index[slicing_dim] = 10 + start_index
    global_shape_t = [10, 10, 10]
    global_shape_t[slicing_dim] = 30
    global_shape = make_3d_shape_from_shape(global_shape_t)
    chunk_shape_t = [10, 10, 10]
    chunk_shape_t[slicing_dim] += 4  # for padding
    chunk_shape = make_3d_shape_from_shape(chunk_shape_t)
    angles = np.linspace(0, math.pi, global_shape[0], dtype=np.float32)
    block = DataSetBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
        slicing_dim=slicing_dim,
        block_start=start_index,
        chunk_start=10,
        global_shape=global_shape,
        chunk_shape=chunk_shape,
        padding=(2, 2),
    )

    assert block.is_padded is True
    assert block.padding == (2, 2)
    assert block.is_last_in_chunk is (boundary == "after")


@pytest.mark.parametrize("slicing_dim", [0, 1, 2])
@pytest.mark.parametrize("boundary", ["before", "after"])
def test_partial_block_with_padding_global_boundaries(
    slicing_dim: Literal[0, 1, 2], boundary: Literal["before", "after"]
):
    block_shape = [10, 10, 10]
    block_shape[slicing_dim] = 6
    padding = (2, 2)
    start_index = -padding[0] if boundary == "before" else 6
    data = np.ones(block_shape, dtype=np.float32)
    chunk_shape_t = [10, 10, 10]
    chunk_shape_t[slicing_dim] += padding[0] + padding[1]  # for padding
    chunk_shape = make_3d_shape_from_shape(chunk_shape_t)
    global_index = [0, 0, 0]
    global_index[slicing_dim] = (
        -padding[0]
        if boundary == "before"
        else 30 - block_shape[slicing_dim] + padding[1]
    )
    chunk_start = -padding[0] if boundary == "before" else 20 - padding[0]
    global_shape_t = [10, 10, 10]
    global_shape_t[slicing_dim] = 30
    global_shape = make_3d_shape_from_shape(global_shape_t)
    angles = np.linspace(0, math.pi, global_shape[0], dtype=np.float32)
    block = DataSetBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
        slicing_dim=slicing_dim,
        block_start=start_index,
        chunk_start=chunk_start,
        global_shape=global_shape,
        chunk_shape=chunk_shape,
        padding=padding,
    )

    assert block.is_padded is True
    assert block.padding == padding
    assert block.global_index == tuple(global_index)
    assert block.is_last_in_chunk is (boundary == "after")


# block_shape <= chunk_shape
@pytest.mark.parametrize("block_shape", [(11, 10, 10), (10, 11, 10), (10, 10, 11)])
def test_inconsistent_block_shape(block_shape: Tuple[int, int, int]):
    data = np.ones(block_shape, dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
            slicing_dim=0,
            block_start=0,
            chunk_start=0,
            global_shape=(10, 10, 10),
            chunk_shape=(10, 10, 10),
        )


@pytest.mark.parametrize(
    "block_shape, slicing_dim",
    [
        ((10, 5, 10), 0),
        ((10, 10, 5), 0),
        ((10, 5, 5), 0),
        ((5, 10, 10), 1),
        ((10, 10, 5), 1),
        ((5, 10, 5), 1),
        ((5, 10, 10), 2),
        ((10, 5, 10), 2),
        ((5, 5, 10), 2),
    ],
    ids=[
        "slice_dim_0_dim_1",
        "slice_dim_0_dim_2",
        "slice_dim_0_dim_1_2",
        "slice_dim_1_dim_0",
        "slice_dim_1_dim_2",
        "slice_dim_1_dim_0_2",
        "slice_dim_2_dim_0",
        "slice_dim_2_dim_1",
        "slice_dim_2_dim_0_1",
    ],
)
def test_inconsistent_non_slicing_dims_block_global(
    block_shape: Tuple[int, int, int], slicing_dim: Literal[0, 1, 2]
):
    data = np.ones(block_shape, dtype=np.float32)
    angles = np.linspace(0, math.pi, block_shape[0], dtype=np.float32)
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
            slicing_dim=slicing_dim,
            block_start=0,
            chunk_start=0,
            global_shape=(10, 10, 10),
            chunk_shape=(10, 10, 10),
        )


# global and chunk shape in non-slicing dim must be the same
@pytest.mark.parametrize("slicing_dim", [0, 1, 2])
def test_inconsistent_global_chunk_shape_non_slice_dim(slicing_dim: Literal[0, 1, 2]):
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    global_shape = [12, 12, 12]
    global_shape[slicing_dim] = 10
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
            slicing_dim=slicing_dim,
            block_start=0,
            chunk_start=0,
            global_shape=make_3d_shape_from_shape(global_shape),
            chunk_shape=(10, 10, 10),
        )


# chunk_index outside of chunk_shape
@pytest.mark.parametrize(
    "block_start",
    [-1, 10, 50],
)
def test_inconsistent_block_start(block_start: int):
    block_shape = [10, 10, 10]
    data = np.ones(block_shape, dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
            slicing_dim=0,
            block_start=block_start,
            chunk_start=0,
            global_shape=(10, 10, 10),
            chunk_shape=(10, 10, 10),
        )


# chunk_index + block_shape not exceed chunk_shape (proxied by start + slicing dim)
def test_inconsistent_block_span():
    block_shape = [10, 10, 10]
    data = np.ones(block_shape, dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
            slicing_dim=0,
            block_start=1,
            chunk_start=0,
            global_shape=(10, 10, 10),
            chunk_shape=(10, 10, 10),
        )


# chunk_shape <= global_shape
@pytest.mark.parametrize("chunk_shape", [(11, 10, 10), (10, 11, 10), (10, 10, 11)])
def test_inconsistent_chunk_shape(chunk_shape: Tuple[int, int, int]):
    block_shape = [10, 10, 10]
    data = np.ones(block_shape, dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
            slicing_dim=0,
            block_start=0,
            chunk_start=0,
            global_shape=(10, 10, 10),
            chunk_shape=(11, 10, 10),
        )


# global_index + chunk_shape not exceed global_shape
def test_inconsistent_chunk_span():
    block_shape = [10, 10, 10]
    data = np.ones(block_shape, dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
            slicing_dim=0,
            block_start=0,
            chunk_start=1,
            global_shape=(10, 10, 10),
            chunk_shape=(10, 10, 10),
        )


# global_index outside of global_shape
@pytest.mark.parametrize(
    "chunk_start",
    [-1, 10, 50],
)
def test_inconsistent_chunk_start(chunk_start: int):
    block_shape = [10, 10, 10]
    data = np.ones(block_shape, dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
            slicing_dim=0,
            block_start=0,
            chunk_start=chunk_start,
            global_shape=(10, 10, 10),
            chunk_shape=(10, 10, 10),
        )


def test_modify_data_no_shape_change():
    # `DataSetBlock` overrides `data.setter` inherited from `BaseBlock` so should be tested,
    # even though this test looks very similar to analagous test for `BaseBlock.data.setter`
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 30, dtype=np.float32)
    block = DataSetBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
        block_start=0,
        chunk_start=10,
        slicing_dim=0,
        global_shape=(30, 10, 10),
    )

    block.data = 2 * data

    np.testing.assert_array_equal(block.data, 2 * data)


def test_modify_data_shape_change():
    # `DataSetBlock` overrides `data.setter` inherited from `BaseBlock` so should be tested,
    # even though this test looks very similar to analagous test for `BaseBlock.data.setter`
    block_start = 2
    chunk_start = 10
    data = np.ones((5, 10, 10), dtype=np.float32)
    global_shape = (30, 10, 10)
    angles = np.linspace(0, math.pi, global_shape[0], dtype=np.float32)
    block = DataSetBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
        block_start=block_start,
        chunk_start=chunk_start,
        slicing_dim=0,
        global_shape=global_shape,
        chunk_shape=(10, 10, 10),
    )

    block.data = 2 * np.ones((5, 30, 20), dtype=np.float32)

    np.testing.assert_array_equal(block.data, 2.0)
    assert block.shape == (5, 30, 20)
    assert block.global_shape == (global_shape[0], 30, 20)
    assert block.chunk_shape == (10, 30, 20)
    assert block.chunk_index == (block_start, 0, 0)
    assert block.global_index == (chunk_start + block_start, 0, 0)


@pytest.mark.parametrize("new_size", [3, 7])
def test_modify_data_shape_change_in_slicing_dim_fails(new_size: int):
    block_start = 2
    chunk_start = 10
    data = np.ones((5, 10, 10), dtype=np.float32)
    global_shape = (30, 10, 10)
    angles = np.linspace(0, math.pi, global_shape[0], dtype=np.float32)
    block = DataSetBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
        block_start=block_start,
        chunk_start=chunk_start,
        slicing_dim=0,
        global_shape=global_shape,
        chunk_shape=(10, 10, 10),
    )

    with pytest.raises(ValueError):
        block.data = 2 * np.ones((new_size, 10, 10), dtype=np.float32)
