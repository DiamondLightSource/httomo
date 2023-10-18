import numpy as np
import pytest
from httomo.utils import xp, gpu_enabled
from httomo.dataset import DataSet


@pytest.fixture
def dataset():
    data = np.ones((10, 10, 10), dtype=np.float32)
    flats = 2 * np.ones((10, 10), dtype=np.float32)
    darks = 3 * np.ones((10, 10), dtype=np.float32)
    angles = 4 * np.ones((20,), dtype=np.float32)

    return DataSet(data, angles, flats, darks)


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


@pytest.mark.skipif(not gpu_enabled, reason="skipped as cupy is not available")
@pytest.mark.cupy
def test_gpu_transfer_and_back(dataset):
    with xp.cuda.Device(0):
        dataset.to_gpu()

        assert dataset.data.device.id == 0
        assert dataset.is_gpu is True
        assert dataset.flats.device.id == 0
        assert dataset.angles.device.id == 0
        assert dataset.darks.device.id == 0
        assert dataset.angles_radians.device.id == 0

        dataset.to_cpu()

        assert dataset.is_gpu is False


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
@pytest.mark.parametrize(
    "field", ["darks", "flats", "angles", "angles_radians", "dark", "flat"]
)
def test_reset_field_gpu(dataset, field):
    original_value = getattr(dataset, field).ravel()[0]
    with xp.cuda.Device(0):
        dataset.to_gpu()
        dataset.unlock()
        setattr(dataset, field, 2 * getattr(dataset, field))  # dataset.darks = 2 * dataset.darks
        dataset.lock()

        xp.testing.assert_array_equal(getattr(dataset, field), 2 * original_value)
        assert dataset.is_gpu is True
        assert getattr(dataset, field).device.id == 0

    dataset.to_cpu()
    np.testing.assert_array_equal(getattr(dataset, field), 2 * original_value)
    
    # dataset.to_gpu()
    # dataset.unlock()
    # dataset.darks = 2 * dataset.darks
    # dataset.lock()
    # dataset.to_cpu()
    # assert dataset.darks == 2 * original_value


@pytest.mark.skipif(not gpu_enabled, reason="skipped as cupy is not available")
@pytest.mark.cupy
@pytest.mark.parametrize(
    "field", ["darks", "flats", "angles", "angles_radians", "dark", "flat"]
)
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
def test_reset_field_numpy(dataset, field):
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
