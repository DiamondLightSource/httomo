import os
from typing import Union
from mpi4py import MPI
import pytest
from pytest_mock import MockerFixture
import numpy as np

from httomo.data.hdf.loaders import LoaderData
from httomo.runner.dataset import DataSet
from httomo.runner.loader import make_loader
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
                flats=2 * np.ones((10, 10), dtype=np.float32),
                darks=3 * np.ones((10, 10), dtype=np.float32),
                angles=4 * np.ones((10, 10), dtype=np.float32),
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

    np.testing.assert_array_equal(dataset.data, 1.0)
    np.testing.assert_array_equal(dataset.flats, 2.0)
    np.testing.assert_array_equal(dataset.darks, 3.0)
    np.testing.assert_array_equal(dataset.angles, 4.0)
