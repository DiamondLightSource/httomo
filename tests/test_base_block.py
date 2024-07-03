import math

import pytest
import numpy as np
from pytest_mock import MockerFixture

from httomo.base_block import BaseBlock
from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.utils import gpu_enabled, xp


def test_longer_angles_length_half_projections():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, data.shape[0] // 2, dtype=np.float32)
    block = BaseBlock(data, aux_data=AuxiliaryData(angles=angles))
    assert len(block.angles) == data.shape[0] // 2


def test_modify_data_no_shape_change():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 30, dtype=np.float32)
    block = BaseBlock(
        data=data,
        aux_data=AuxiliaryData(angles=angles),
    )

    block.data = 2 * data

    np.testing.assert_array_equal(block.data, 2 * data)


def test_modify_data_shape_change():
    data = np.ones((5, 10, 10), dtype=np.float32)
    global_shape = (30, 10, 10)
    angles = np.linspace(0, math.pi, global_shape[0], dtype=np.float32)
    block = BaseBlock(data=data, aux_data=AuxiliaryData(angles=angles))

    block.data = 2 * np.ones((5, 30, 20), dtype=np.float32)

    np.testing.assert_array_equal(block.data, 2.0)
    assert block.shape == (5, 30, 20)


# flats / darks  get/set in the aux object + change is observable in aux (persistent)


def test_darks_and_flats_get():
    data = np.ones((10, 10, 10), dtype=np.float32)
    darks = 2.0 * np.ones((3, 10, 10), dtype=np.float32)
    flats = 3.0 * np.ones((5, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = BaseBlock(
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
    block = BaseBlock(data=data, aux_data=aux_data)

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
    block = BaseBlock(data=data, aux_data=aux_data)

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
    block = BaseBlock(data=data, aux_data=aux_data)

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
    block = BaseBlock(data=data, aux_data=aux_data)

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
    block = BaseBlock(data=data, aux_data=aux_data)

    block.angles = np.ones(10, dtype=np.float32)

    np.testing.assert_array_equal(aux_data.get_angles(), 1.0)


def test_angles_set_via_alias():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    aux_data = AuxiliaryData(angles=angles)
    block = BaseBlock(data=data, aux_data=aux_data)

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
    mocker.patch("httomo.base_block.gpu_enabled", False)
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = BaseBlock(data=data, aux_data=AuxiliaryData(angles=angles))
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
    block = BaseBlock(data=data, aux_data=AuxiliaryData(angles=angles))

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
    block = BaseBlock(data=data, aux_data=AuxiliaryData(angles=angles))

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
    block = BaseBlock(data=data, aux_data=AuxiliaryData(angles=angles))
    block.to_gpu()

    block.to_cpu()

    assert block.is_gpu is False
    assert block.is_cpu is True
    assert getattr(block.data, "device", None) is None


def test_transfer_to_cpu_with_no_gpu(mocker: MockerFixture):
    mocker.patch("httomo.base_block.gpu_enabled", False)
    mocker.patch("httomo.base_block.xp", np)

    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = BaseBlock(data=data, aux_data=AuxiliaryData(angles=angles))

    block.to_cpu()

    assert block.is_gpu is False
    assert block.is_cpu is True
    assert getattr(block.data, "device", None) is None


def test_to_cpu_twice_has_no_effect():
    data = np.ones((10, 10, 10), dtype=np.float32)
    angles = np.linspace(0, math.pi, 10, dtype=np.float32)
    block = BaseBlock(data=data, aux_data=AuxiliaryData(angles=angles))

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
    block = BaseBlock(data=data, aux_data=aux_data)

    block.to_gpu()

    assert block.darks.device == xp.cuda.Device()
    assert block.flats.device == xp.cuda.Device()


def test_attributes_array(dummy_block: BaseBlock):
    expected = set(
        ["data", "flats", "darks", "angles", "angles_radians", "dark", "flat"]
    )
    actual = set(dir(dummy_block))
    assert actual == expected
