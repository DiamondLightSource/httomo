import os
from pathlib import Path
from typing import Union

from mpi4py import MPI
import pytest
from pytest_mock import MockerFixture
import numpy as np

from httomo.data.hdf.loaders import LoaderData
from httomo.runner.dataset import DataSet
from httomo.runner.loader import StandardTomoLoader, make_loader
from .testing_utils import make_mock_repo


def test_loader_execute_raises_exception(mocker: MockerFixture, dummy_dataset: DataSet):
    class FakeModule:
        def fake_loader(name) -> LoaderData:
            return LoaderData()

    mocker.patch("importlib.import_module", return_value=FakeModule)
    loader = make_loader(
        make_mock_repo(mocker), "mocked_module_path", "fake_loader", MPI.COMM_WORLD
    )
    with pytest.raises(NotImplementedError):
        loader.execute(dummy_dataset)


def test_loader_load_produces_dataset(mocker: MockerFixture):
    class FakeModule:
        def fake_loader(name, in_file: Union[os.PathLike, str]) -> LoaderData:
            assert name == "dataset"
            assert in_file == "some_test_file"
            return LoaderData(
                data=np.ones((10, 10, 10), dtype=np.float32),
                flats=2 * np.ones((2, 10, 10), dtype=np.float32),
                darks=3 * np.ones((2, 10, 10), dtype=np.float32),
                angles=4 * np.ones((10,), dtype=np.float32),
                angles_total=10,
                detector_x=5,
                detector_y=14,
            )

    mocker.patch("importlib.import_module", return_value=FakeModule)
    loader = make_loader(
        make_mock_repo(mocker),
        "mocked_module_path",
        "fake_loader",
        MPI.COMM_WORLD,
        name="dataset",
        in_file="some_test_file",
    )
    dataset = loader.load()

    assert loader.detector_x == 5
    assert loader.detector_y == 14
    assert dataset.global_index == (0, 0, 0)
    assert dataset.global_shape == (10, 10, 10)
    np.testing.assert_array_equal(dataset.data, 1.0)
    np.testing.assert_array_equal(dataset.flats, 2.0)
    np.testing.assert_array_equal(dataset.darks, 3.0)
    np.testing.assert_array_equal(dataset.angles, 4.0)


def test_standard_tomo_loader_get_slicing_dim(
    standard_data: str,
    standard_data_path: str,
):
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    assert loader.slicing_dim == SLICING_DIM


def test_standard_tomo_loader_get_global_shape(
    standard_data: str,
    standard_data_path: str,
):
    SLICING_DIM = 0
    GLOBAL_DATA_SHAPE = (220, 128, 160)
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    assert loader.global_shape == GLOBAL_DATA_SHAPE
