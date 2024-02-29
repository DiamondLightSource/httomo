import math
from typing import Tuple
import numpy as np
from pytest_mock import MockerFixture
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.utils import gpu_enabled, xp
import pytest

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


@pytest.mark.parametrize("slicing_dim", [0, 1, 2])
def test_full_block_for_chunked_data(slicing_dim: int):
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
def test_partial_block_for_chunked_data(slicing_dim: int, last_in_chunk: bool):
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
    block_shape: Tuple[int, int, int], slicing_dim: int
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
def test_inconsistent_global_chunk_shape_non_slice_dim(slicing_dim: int):
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


# angles length >= block.global_shape[0]
def test_inconsistent_angles_length():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, data.shape[0] - 2, dtype=np.float32)
    with pytest.raises(ValueError):
        DataSetBlock(
            data,
            aux_data=AuxiliaryData(angles=angles),
        )


def test_longer_angles_length_than_projections():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, data.shape[0] + 2, dtype=np.float32)
    block = DataSetBlock(
        data,
        aux_data=AuxiliaryData(angles=angles),
    )
    assert len(block.angles) == data.shape[0] + 2


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


# flats / darks  get/set in the aux object + change is observable in aux (persistent)


def test_darks_and_flats_get():
    data = np.ones((10, 10, 10), dtype=np.float32)
    darks = 2.0 * np.ones((3, 10, 10), dtype=np.float32)
    flats = 3.0 * np.ones((5, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(
        data=data, aux_data=AuxiliaryData(angles=angles, darks=darks, flats=flats)
    )

    np.testing.assert_array_equal(darks, block.darks)
    np.testing.assert_array_equal(darks, block.dark)
    np.testing.assert_array_equal(flats, block.flats)
    np.testing.assert_array_equal(flats, block.flat)


def test_darks_and_flats_set_same_shape():
    data = np.ones((10, 10, 10), dtype=np.float32)
    darks = 2.0 * np.ones((3, 10, 10), dtype=np.float32)
    flats = 3.0 * np.ones((5, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles, darks=darks, flats=flats)
    block = DataSetBlock(data=data, aux_data=aux_data)

    block.darks = block.darks * 2
    block.flats = block.flats * 2

    np.testing.assert_array_equal(darks * 2, block.darks)
    np.testing.assert_array_equal(flats * 2, block.flats)
    np.testing.assert_array_equal(darks * 2, aux_data.get_darks())
    np.testing.assert_array_equal(flats * 2, aux_data.get_flats())


def test_darks_and_flats_set_different_dtype():
    data = np.ones((10, 10, 10), dtype=np.float32)
    darks = 2 * np.ones((3, 10, 10), dtype=np.uint16)
    flats = 3 * np.ones((5, 10, 10), dtype=np.uint16)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles, darks=darks, flats=flats)
    block = DataSetBlock(data=data, aux_data=aux_data)

    block.darks = block.darks.astype(np.float32)
    block.flats = block.flats.astype(np.float32)

    assert block.darks.dtype == np.float32
    assert block.flats.dtype == np.float32
    assert aux_data.get_darks().dtype == np.float32
    assert aux_data.get_flats().dtype == np.float32


def test_darks_and_flats_set_different_shape():
    data = np.ones((10, 10, 10), dtype=np.float32)
    darks = 2.0 * np.ones((3, 10, 10), dtype=np.float32)
    flats = 3.0 * np.ones((5, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles, darks=darks, flats=flats)
    block = DataSetBlock(data=data, aux_data=aux_data)

    block.darks = np.mean(block.darks, axis=0, keepdims=True)
    block.flats = np.mean(block.flats, axis=0, keepdims=True)

    np.testing.assert_array_equal(
        block.darks, 2.0 * np.ones((1, 10, 10), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        block.dark, 2.0 * np.ones((1, 10, 10), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        block.flats, 3.0 * np.ones((1, 10, 10), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        block.flat, 3.0 * np.ones((1, 10, 10), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        aux_data.get_darks(), 2.0 * np.ones((1, 10, 10), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        aux_data.get_flats(), 3.0 * np.ones((1, 10, 10), dtype=np.float32)
    )


def test_darks_and_flats_set_via_alias():
    data = np.ones((10, 10, 10), dtype=np.float32)
    darks = 2.0 * np.ones((3, 10, 10), dtype=np.float32)
    flats = 3.0 * np.ones((5, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles, darks=darks, flats=flats)
    block = DataSetBlock(data=data, aux_data=aux_data)

    block.dark = np.mean(block.darks, axis=0, keepdims=True)
    block.flat = np.mean(block.flats, axis=0, keepdims=True)

    np.testing.assert_array_equal(
        aux_data.get_darks(), 2.0 * np.ones((1, 10, 10), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        aux_data.get_flats(), 3.0 * np.ones((1, 10, 10), dtype=np.float32)
    )


def test_angles_set():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles)
    block = DataSetBlock(data=data, aux_data=aux_data)

    block.angles = np.ones(10, dtype=np.float32)

    np.testing.assert_array_equal(aux_data.get_angles(), 1.0)


def test_angles_set_via_alias():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles)
    block = DataSetBlock(data=data, aux_data=aux_data)

    block.angles_radians = np.ones(10, dtype=np.float32)

    np.testing.assert_array_equal(aux_data.get_angles(), 1.0)


def test_aux_can_drop_darks_flats():
    # TODO: consider a separate test suite for AuxiliaryData
    darks = 2.0 * np.ones((3, 10, 10), dtype=np.float32)
    flats = 3.0 * np.ones((5, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles, darks=darks, flats=flats)

    assert aux_data.darks_dtype == np.float32
    assert aux_data.flats_dtype == np.float32
    assert aux_data.darks_shape == darks.shape
    assert aux_data.flats_shape == flats.shape
    assert aux_data.angles_dtype == np.float32
    assert aux_data.angles_length == 10

    aux_data.drop_darks_flats()

    assert aux_data.get_darks() is None
    assert aux_data.get_flats() is None
    assert aux_data.darks_dtype is None
    assert aux_data.flats_dtype is None
    assert aux_data.darks_shape == (0, 0, 0)
    assert aux_data.flats_shape == (0, 0, 0)
    assert aux_data.angles_dtype == np.float32
    assert aux_data.angles_length == 10


# to_gpu, to_cpu (like DataSet) -> should not transfer flats/darks immediately


# to_gpu when no GPU available should throw
def test_to_gpu_fails_when_gpu_not_enabled(mocker):
    mocker.patch("httomo.runner.dataset.gpu_enabled", False)
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(data=data, aux_data=AuxiliaryData(angles=angles))
    with pytest.raises(ValueError) as e:
        block.to_gpu()

    assert "no GPU available" in str(e)


# to_gpu with enabled GPU transfers the data
@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
def test_to_gpu_transfers_the_data():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(data=data, aux_data=AuxiliaryData(angles=angles))

    block.to_gpu()

    assert block.is_gpu is True
    assert block.is_cpu is False
    assert block.data.device == xp.cuda.Device()


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
def test_to_gpu_twice_has_no_effect():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(data=data, aux_data=AuxiliaryData(angles=angles))

    block.to_gpu()
    # this gives us the memory address (pointer) to the data on GPU
    # since we want to make sure that no copy is taken on the second call
    address = block.data.__cuda_array_interface__["data"]
    block.to_gpu()

    assert block.data.__cuda_array_interface__["data"] == address


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
def test_to_cpu_transfers_the_data_back():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(data=data, aux_data=AuxiliaryData(angles=angles))
    block.to_gpu()

    block.to_cpu()

    assert block.is_gpu is False
    assert block.is_cpu is True
    assert getattr(block.data, "device", None) is None


def test_transfer_to_cpu_with_no_gpu(mocker: MockerFixture):
    mocker.patch("httomo.runner.dataset.gpu_enabled", False)
    mocker.patch("httomo.runner.dataset.xp", np)

    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(data=data, aux_data=AuxiliaryData(angles=angles))

    block.to_cpu()

    assert block.is_gpu is False
    assert block.is_cpu is True
    assert getattr(block.data, "device", None) is None


def test_to_cpu_twice_has_no_effect():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = DataSetBlock(data=data, aux_data=AuxiliaryData(angles=angles))

    # this gives us the memory address (pointer) to the data on CPU
    # since we want to make sure that no copy is taken on the second call
    address = block.data.__array_interface__["data"]
    block.to_cpu()

    assert block.data.__array_interface__["data"] == address


# flats / darks should be returned on the same devices as where `data` is
@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
def test_returns_flats_on_gpu_when_the_data_is_there():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    darks = 2.0 * np.ones((3, 10, 10), dtype=np.float32)
    flats = 3.0 * np.ones((5, 10, 10), dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles, darks=darks, flats=flats)
    block = DataSetBlock(data=data, aux_data=aux_data)

    block.to_gpu()

    assert block.darks.device == xp.cuda.Device()
    assert block.flats.device == xp.cuda.Device()
