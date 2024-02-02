from pathlib import Path
from httomo.methods import calculate_stats, save_intermediate_data

import numpy as np
import pytest
from mpi4py import MPI
import h5py

from httomo.utils import gpu_enabled, xp
from httomo.runner.dataset import DataSet


def test_calculate_stats_simple():
    data = np.arange(30, dtype=np.float32).reshape((2, 3, 5)) - 10.0
    ret = calculate_stats(data)

    assert ret == (-10.0, 19.0, np.sum(data), 30)


def test_calculate_stats_with_nan_and_inf():
    data = np.arange(30, dtype=np.float32).reshape((2, 3, 5)) - 10.0
    expected = np.sum(data) - data[1, 1, 1] - data[0, 2, 3] - data[1, 2, 3]
    data[1, 1, 1] = float("inf")
    data[0, 2, 3] = float("-inf")
    data[1, 2, 3] = float("nan")
    ret = calculate_stats(data)

    assert ret == (-10.0, 19.0, expected, 30)


@pytest.mark.skipif(
    not gpu_enabled or xp.cuda.runtime.getDeviceCount() == 0,
    reason="skipped as cupy is not available",
)
@pytest.mark.cupy
def test_calculate_status_gpu():
    data = xp.arange(30, dtype=np.float32).reshape((2, 3, 5)) - 10.0
    ret = calculate_stats(data)

    assert ret == (-10.0, 19.0, np.sum(data.get()), 30)


@pytest.mark.parametrize("gpu", [False, True])
def test_save_intermediate_data(dummy_dataset: DataSet, tmp_path: Path, gpu: bool):
    if gpu and not gpu_enabled:
        pytest.skip("No GPU available")

    # use increasing numbers in the data, to make sure blocks have different content
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )

    if gpu:
        dummy_dataset.to_gpu()

    with h5py.File(
        tmp_path / "test_file.h5", "w", driver="mpio", comm=MPI.COMM_WORLD
    ) as file:
        # save in 2 blocks, starting with the second to confirm order-independence
        save_intermediate_data(
            dummy_dataset.make_block(0, 3),
            file,
            path="/data",
            detector_x=10,
            detector_y=20,
        )
        save_intermediate_data(
            dummy_dataset.make_block(0, 0, 3),
            file,
            path="/data",
            detector_x=10,
            detector_y=20,
        )

    if gpu:
        assert dummy_dataset.is_gpu
        dummy_dataset.to_cpu()

    with h5py.File(tmp_path / "test_file.h5", "r") as file:
        assert "/data" in file
        data = file["/data"]
        np.testing.assert_array_equal(data, dummy_dataset.data)
        assert "/angles" in file
        angles = file["/angles"]
        np.testing.assert_array_equal(angles, dummy_dataset.angles)
        assert "test_file.h5" in file
        np.testing.assert_array_equal(file["test_file.h5"], [0, 0])
        assert "data_dims" in file
        np.testing.assert_array_equal(file["data_dims"]["detector_x_y"], [10, 20])


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_save_intermediate_data_mpi(dummy_dataset: DataSet, tmp_path: Path):
    comm = MPI.COMM_WORLD
    # make sure we use the same tmp_path on both processes
    tmp_path = comm.bcast(tmp_path)
    # use increasing numbers in the data, to make sure blocks have different content
    dummy_dataset.data = np.arange(dummy_dataset.data.size, dtype=np.float32).reshape(
        dummy_dataset.shape
    )
    # give each process only a portion of the data
    dataset = DataSet(
        data=dummy_dataset.data[0:5, :, :]
        if comm.rank == 0
        else dummy_dataset.data[5:, :, :],
        angles=dummy_dataset.angles,
        flats=dummy_dataset.flats,
        darks=dummy_dataset.darks,
        global_shape=dummy_dataset.shape,
        global_index=(0, 0, 0) if comm.rank == 0 else (5, 0, 0),
    )

    with h5py.File(tmp_path / "test_file.h5", "w", driver="mpio", comm=comm) as file:
        # save in 2 blocks, starting with the second to confirm order-independence
        save_intermediate_data(
            dataset.make_block(0, 3),
            file,
            path="/data",
            detector_x=10,
            detector_y=20,
        )
        save_intermediate_data(
            dataset.make_block(0, 0, 3),
            file,
            path="/data",
            detector_x=10,
            detector_y=20,
        )

    # open without MPI, looking at the full file
    with h5py.File(tmp_path / "test_file.h5", "r") as file:
        assert "/data" in file
        data = file["/data"]
        np.testing.assert_array_equal(data, dummy_dataset.data)
        assert "/angles" in file
        angles = file["/angles"]
        np.testing.assert_array_equal(angles, dummy_dataset.angles)
        assert "test_file.h5" in file
        np.testing.assert_array_equal(file["test_file.h5"], [0, 0])
        assert "data_dims" in file
        np.testing.assert_array_equal(file["data_dims"]["detector_x_y"], [10, 20])
