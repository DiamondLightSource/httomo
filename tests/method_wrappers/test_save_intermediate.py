from pathlib import Path
from typing import Tuple
import pytest
from pytest_mock import MockerFixture
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.save_intermediate import SaveIntermediateFilesWrapper

from httomo.utils import gpu_enabled
from httomo.runner.dataset import DataSetBlock
import h5py
from mpi4py import MPI
from httomo.runner.loader import LoaderInterface
from httomo.runner.method_wrapper import MethodWrapper
from ..testing_utils import make_mock_repo
import httomo
import numpy as np


@pytest.mark.cupy
def test_save_intermediate(
    mocker: MockerFixture, dummy_block: DataSetBlock, tmp_path: Path
):
    loader: LoaderInterface = mocker.create_autospec(
        LoaderInterface, instance=True, detector_x=10, detector_y=20
    )

    class FakeModule:
        def save_intermediate_data(
            data,
            global_shape: Tuple[int, int, int],
            global_index: Tuple[int, int, int],
            slicing_dim: int,
            file: h5py.File,
            path: str,
            detector_x: int,
            detector_y: int,
            angles: np.ndarray,
        ):
            assert isinstance(data, np.ndarray)
            assert data.shape == dummy_block.shape
            assert global_index == (0, 0, 0)
            assert global_shape == dummy_block.shape
            assert slicing_dim == 0
            assert Path(file.filename).name == "task1-testpackage-testmethod-XXX.h5"
            assert detector_x == 10
            assert detector_y == 20
            assert path == "/data"

    mocker.patch("importlib.import_module", return_value=FakeModule)
    prev_method = mocker.create_autospec(
        MethodWrapper,
        instance=True,
        task_id="task1",
        package_name="testpackage",
        method_name="testmethod",
        recon_algorithm="XXX",
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "httomo.methods",
        "save_intermediate_data",
        MPI.COMM_WORLD,
        loader=loader,
        out_dir=tmp_path,
        prev_method=prev_method,
    )
    assert isinstance(wrp, SaveIntermediateFilesWrapper)
    res = wrp.execute(dummy_block)

    assert res == dummy_block


@pytest.mark.cupy
def test_save_intermediate_defaults_out_dir(mocker: MockerFixture, tmp_path: Path):
    loader: LoaderInterface = mocker.create_autospec(
        LoaderInterface, instance=True, detector_x=10, detector_y=20
    )

    class FakeModule:
        def save_intermediate_data(
            data,
            global_shape: Tuple[int, int, int],
            global_index: Tuple[int, int, int],
            slicing_dim: int,
            file: h5py.File,
            path: str,
            detector_x: int,
            detector_y: int,
            angles: np.ndarray,
        ):
            pass

    mocker.patch("importlib.import_module", return_value=FakeModule)
    prev_method = mocker.create_autospec(
        MethodWrapper,
        instance=True,
        task_id="task1",
        package_name="testpackage",
        method_name="testmethod",
        recon_algorithm="XXX",
    )
    mocker.patch.object(httomo.globals, "run_out_dir", tmp_path)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "httomo.methods",
        "save_intermediate_data",
        MPI.COMM_WORLD,
        loader=loader,
        prev_method=prev_method,
    )
    assert isinstance(wrp, SaveIntermediateFilesWrapper)
    assert wrp._file.filename.startswith(str(tmp_path))


@pytest.mark.parametrize("gpu", [False, True], ids=["CPU", "GPU"])
def test_save_intermediate_leaves_gpu_data(
    mocker: MockerFixture, dummy_block: DataSetBlock, tmp_path: Path, gpu: bool
):
    if gpu and not gpu_enabled:
        pytest.skip("No GPU available")

    loader: LoaderInterface = mocker.create_autospec(
        LoaderInterface, instance=True, detector_x=10, detector_y=20
    )

    class FakeModule:
        def save_intermediate_data(
            data,
            global_shape: Tuple[int, int, int],
            global_index: Tuple[int, int, int],
            slicing_dim: int,
            file: h5py.File,
            path: str,
            detector_x: int,
            detector_y: int,
            angles: np.ndarray,
        ):
            assert isinstance(data, np.ndarray)
            assert getattr(data, "device", None) is None

    mocker.patch("importlib.import_module", return_value=FakeModule)
    prev_method = mocker.create_autospec(
        MethodWrapper,
        instance=True,
        task_id="task1",
        package_name="testpackage",
        method_name="testmethod",
        recon_algorithm="XXX",
    )
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy" if gpu else "cpu"),
        "httomo.methods",
        "save_intermediate_data",
        MPI.COMM_WORLD,
        loader=loader,
        out_dir=tmp_path,
        prev_method=prev_method,
    )

    if gpu is True:
        dummy_block.to_gpu()

    assert dummy_block.is_gpu == gpu
    res = wrp.execute(dummy_block)

    assert res.is_gpu == gpu
