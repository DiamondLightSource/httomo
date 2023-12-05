import numpy as np
import pytest
from pytest_mock import MockerFixture
from httomo.utils import xp, gpu_enabled
from httomo.runner.dataset import DataSet


@pytest.fixture
def dataset():
    data = np.ones((10, 10, 10), dtype=np.float32)
    flats = 2 * np.ones((2, 10, 10), dtype=np.float32)
    darks = 3 * np.ones((2, 10, 10), dtype=np.float32)
    angles = 4 * np.ones((20,), dtype=np.float32)

    return DataSet(data, angles, flats, darks)


@pytest.fixture
def disable_gpu(mocker: MockerFixture):
    gpu_enabled = mocker.patch("httomo.runner.dataset.gpu_enabled", return_value=False)
    gpu_enabled.__bool__.return_value = False
    return gpu_enabled


def test_works_with_numpy(dataset):
    np.testing.assert_array_equal(dataset.data, 1)
    np.testing.assert_array_equal(dataset.flats, 2)
    np.testing.assert_array_equal(dataset.flat, 2)
    np.testing.assert_array_equal(dataset.darks, 3)
    np.testing.assert_array_equal(dataset.dark, 3)
    np.testing.assert_array_equal(dataset.angles, 4)
    np.testing.assert_array_equal(dataset.angles_radians, 4)
    assert dataset.is_gpu is False
    assert dataset.is_locked is True


def test_can_get_shape_info(dataset):
    assert dataset.global_shape == dataset.data.shape
    assert dataset.global_index == (0, 0, 0)
    assert dataset.shape == dataset.global_shape


def test_chunked_shape(dataset):
    d2 = DataSet(
        data=dataset.data[2:5, :, :],
        darks=dataset.darks,
        flats=dataset.flats,
        angles=dataset.angles,
        global_shape=dataset.data.shape,
        global_index=(2, 0, 0),
    )

    assert d2.global_shape == dataset.data.shape
    assert d2.global_index == (2, 0, 0)
    assert d2.shape == np.shape(dataset.data[2:5, :, :])


def test_chunked_shape_invalid(dataset):
    with pytest.raises(ValueError) as e:
        DataSet(
            data=dataset.data[2:5, :, :],
            darks=dataset.darks,
            flats=dataset.flats,
            angles=dataset.angles,
            global_shape=(2, 2, 2),
        )
    assert "(2, 2, 2) is incompatible" in str(e)


def test_darks_not_writeable_numpy(dataset):
    with pytest.raises(ValueError):
        dataset.darks[1, 1] = 42.0


def test_flats_not_writeable_numpy(dataset):
    with pytest.raises(ValueError):
        dataset.flats[1, 1] = 42.0


def test_angels_not_writeable_numpy(dataset):
    with pytest.raises(ValueError):
        dataset.angles[1, 1] = 42.0


def test_data_writeable_numpy(dataset):
    dataset.data[1, 1, 1] = 42.0

    assert dataset.data[1, 1, 1] == 42.0


def test_raises_error_on_to_gpu_if_no_gpu_available(dataset: DataSet, disable_gpu):
    with pytest.raises(ValueError) as e:
        dataset.to_gpu()
    assert "cannot transfer to GPU if not enabled" in str(e)


@pytest.mark.skipif(not gpu_enabled, reason="skipped as cupy is not available")
@pytest.mark.cupy
def test_gpu_transfer_and_back(dataset):
    with xp.cuda.Device(0):
        dataset.to_gpu()

        assert dataset.data.device.id == 0
        assert dataset.is_gpu is True
        assert dataset.flats.device.id == 0
        assert dataset.darks.device.id == 0

        dataset.to_cpu()

        assert dataset.is_gpu is False


@pytest.mark.skipif(not gpu_enabled, reason="skipped as cupy is not available")
@pytest.mark.cupy
def test_angles_stay_on_cpu(dataset):
    with xp.cuda.Device(0):
        dataset.to_gpu()

        assert getattr(dataset.angles, "device", None) is None
        assert getattr(dataset.angles_radians, "device", None) is None


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() < 2,
    reason="skipped as cupy and 2 GPUs are not available",
)
@pytest.mark.cupy
def test_gpu_wrong_device(dataset):
    with xp.cuda.Device(0):
        dataset.to_gpu()
        assert dataset.flats.device.id == 0

    with xp.cuda.Device(1):
        with pytest.raises(AssertionError):
            dataset.flats


@pytest.mark.parametrize(
    "field", ["darks", "flats", "angles", "angles_radians", "dark", "flat"]
)
def test_reset_field_while_locked(dataset, field):
    with pytest.raises(ValueError):
        setattr(dataset, field, 2 * getattr(dataset, field))


@pytest.mark.skipif(not gpu_enabled, reason="skipped as cupy is not available")
@pytest.mark.cupy
@pytest.mark.parametrize("field", ["darks", "flats", "dark", "flat"])
def test_reset_field_gpu(dataset, field):
    original_value = getattr(dataset, field).ravel()[0]
    with xp.cuda.Device(0):
        dataset.to_gpu()
        dataset.unlock()
        setattr(
            dataset, field, 2 * getattr(dataset, field)
        )  # dataset.darks = 2 * dataset.darks
        dataset.lock()

        xp.testing.assert_array_equal(getattr(dataset, field), 2 * original_value)
        assert dataset.is_gpu is True
        assert getattr(dataset, field).device.id == 0

    dataset.to_cpu()
    np.testing.assert_array_equal(getattr(dataset, field), 2 * original_value)


@pytest.mark.skipif(not gpu_enabled, reason="skipped as cupy is not available")
@pytest.mark.cupy
@pytest.mark.parametrize("field", ["darks", "flats", "dark", "flat"])
def test_reset_field_on_cpu_with_cached_gpudata(dataset, field):
    original_value = getattr(dataset, field).ravel()[0]
    with xp.cuda.Device(0):
        dataset.to_gpu()
        data = getattr(dataset, field)  # make sure it's transferred
        assert data.device.id == 0
        dataset.to_cpu()

    dataset.unlock()
    setattr(dataset, field, 2 * getattr(dataset, field))
    dataset.lock()
    with xp.cuda.Device(0):
        dataset.to_gpu()
        xp.testing.assert_array_equal(getattr(dataset, field), 2 * original_value)
        assert getattr(dataset, field).device.id == 0


@pytest.mark.parametrize(
    "field", ["darks", "flats", "angles", "angles_radians", "dark", "flat"]
)
@pytest.mark.parametrize("gpu", [True, False])
def test_reset_field_numpy(dataset, field, gpu, request):
    if not gpu:
        request.getfixturevalue(
            "disable_gpu"
        )  # disables GPU conditional on gpu parameter
    dataset.unlock()
    original_value = getattr(dataset, field).ravel()[0]
    setattr(dataset, field, 2 * getattr(dataset, field))
    dataset.unlock()

    np.testing.assert_array_equal(getattr(dataset, field), 2 * original_value)


def test_attributes_array(dataset):
    expected = set(
        ["data", "flats", "darks", "angles", "angles_radians", "dark", "flat"]
    )
    actual = set(dir(dataset))
    assert actual == expected


@pytest.mark.parametrize(
    "dim,start,length", [(0, 0, 3), (0, 3, 2), (1, 0, 4), (1, 3, 1)]
)
def test_can_slice_dataset_to_blocks(
    dataset: DataSet, dim: int, start: int, length: int
):
    # assign increasing numbers
    dataset.data[:] = np.arange(dataset.data.size, dtype=dataset.data.dtype).reshape(
        dataset.data.shape
    )
    block = dataset.make_block(dim, start, length)

    assert block.is_block is True
    assert dataset.is_block is False
    assert block.base.shape == dataset.shape
    assert block.global_shape == dataset.shape
    assert block.chunk_shape == dataset.shape
    assert dataset.chunk_shape == dataset.shape
    assert block.chunk_index == block.global_index
    assert dataset.chunk_index == (0, 0, 0)
    assert dataset.is_last_in_chunk is True
    assert block.is_last_in_chunk is False
    if dim == 0:
        np.testing.assert_array_equal(
            dataset.data[start : start + length, :, :], block.data
        )
        assert block.global_index == (start, 0, 0)
    else:
        np.testing.assert_array_equal(
            dataset.data[:, start : start + length, :], block.data
        )
        assert block.global_index == (0, start, 0)
    # make sure the other fields share the same memory address
    assert (
        block.darks.__array_interface__["data"]
        == dataset.darks.__array_interface__["data"]
    )
    assert (
        block.flats.__array_interface__["data"]
        == dataset.flats.__array_interface__["data"]
    )
    assert (
        block.angles.__array_interface__["data"]
        == dataset.angles.__array_interface__["data"]
    )


def test_is_last_block_in_chunk_property(dataset: DataSet):
    assert dataset.is_last_in_chunk is True
    assert dataset.make_block(0, dataset.shape[0] - 2, 2).is_last_in_chunk is True
    assert dataset.make_block(0, dataset.shape[0] - 4, 2).is_last_in_chunk is False


def test_cannot_slice_a_block(dataset: DataSet):
    block = dataset.make_block(0, 0, 2)
    with pytest.raises(ValueError):
        block.make_block(0, 0, 1)


@pytest.mark.parametrize(
    "field", ["darks", "flats", "angles", "angles_radians", "dark", "flat"]
)
def test_cannot_reset_darks_flats_angles_on_a_block(dataset: DataSet, field: str):
    block = dataset.make_block(0, 0, 2)
    block.unlock()
    dataset.unlock()
    with pytest.raises(ValueError):
        setattr(block, field, np.ones((10, 10), dtype=np.float32))


@pytest.mark.skipif(not gpu_enabled, reason="skipped as cupy is not available")
def test_a_block_reuses_base_cached_gpu_fields(dataset: DataSet):
    with xp.cuda.Device(0):
        dataset.to_gpu()
        darks_orig = dataset.darks
        flats_orig = dataset.flats
        dataset.to_cpu()

        block = dataset.make_block(0, 1, 2)
        block.to_gpu()
        darks_new = block.darks
        flats_new = block.flats

        assert block.is_gpu is True
        # check that they are on the right device
        assert block.data.device.id == 0
        assert block.flats.device.id == 0
        assert block.darks.device.id == 0
        # check that cupy arrays are pointing to the same GPU memory
        assert darks_orig.data == darks_new.data
        assert flats_orig.data == flats_new.data


@pytest.mark.skipif(not gpu_enabled, reason="skipped as cupy is not available")
def test_block_caches_in_base_on_gpu_access(dataset: DataSet):
    with xp.cuda.Device(0):
        block = dataset.make_block(0, 1, 2)
        block.to_gpu()
        darks_new = block.darks

        dataset.to_gpu()
        darks_orig = dataset.darks

        assert darks_orig.data == darks_new.data
