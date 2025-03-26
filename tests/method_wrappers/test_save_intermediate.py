from pathlib import Path
from typing import Callable, List, Optional, Tuple
from unittest import mock
import pytest
from pytest_mock import MockerFixture
from httomo.method_wrappers import make_method_wrapper
from httomo.method_wrappers.save_intermediate import SaveIntermediateFilesWrapper

from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.utils import gpu_enabled, make_3d_shape_from_shape
from httomo.runner.dataset import DataSetBlock
import h5py
from mpi4py import MPI
from httomo.runner.loader import LoaderInterface
from httomo.runner.method_wrapper import MethodWrapper
from ..testing_utils import make_mock_preview_config, make_mock_repo
import httomo
import numpy as np


@pytest.mark.cupy
def test_save_intermediate(
    mocker: MockerFixture, dummy_block: DataSetBlock, tmp_path: Path
):
    FRAMES_PER_CHUNK = 0
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
            frames_per_chunk: int,
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
            assert frames_per_chunk == FRAMES_PER_CHUNK
            assert Path(file.filename).name == "task1-testpackage-testmethod-XXX.h5"
            assert detector_x == 10
            assert detector_y == 20
            assert path == "/data"

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
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
        make_mock_preview_config(mocker),
        loader=loader,
        out_dir=tmp_path,
        prev_method=prev_method,
    )
    assert isinstance(wrp, SaveIntermediateFilesWrapper)
    with mock.patch("httomo.globals.FRAMES_PER_CHUNK", FRAMES_PER_CHUNK):
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
            frames_per_chunk: int,
            path: str,
            detector_x: int,
            detector_y: int,
            angles: np.ndarray,
        ):
            pass

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
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
        make_mock_preview_config(mocker),
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

    FRAMES_PER_CHUNK = 0
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
            frames_per_chunk: int,
            path: str,
            detector_x: int,
            detector_y: int,
            angles: np.ndarray,
        ):
            assert isinstance(data, np.ndarray)
            assert getattr(data, "device", None) is None

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
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
        make_mock_preview_config(mocker),
        loader=loader,
        out_dir=tmp_path,
        prev_method=prev_method,
    )

    if gpu is True:
        dummy_block.to_gpu()

    assert dummy_block.is_gpu == gpu
    with mock.patch("httomo.globals.FRAMES_PER_CHUNK", FRAMES_PER_CHUNK):
        res = wrp.execute(dummy_block)

    assert res.is_gpu == gpu


@pytest.mark.parametrize(
    "padding",
    [(0, 0), (2, 3)],
    ids=["zero-padding", "non-zero-padding"],
)
def test_writes_core_of_blocks_only(
    mocker: MockerFixture, tmp_path: Path, padding: Tuple[int, int]
):
    # Define global data which, for a single process, is equal chunk data
    GLOBAL_SHAPE = (10, 10, 30)
    CHUNK_SHAPE_UNPADDED = GLOBAL_SHAPE
    BLOCK_SHAPE_UNPADDED = (
        CHUNK_SHAPE_UNPADDED[0] // 2,
        CHUNK_SHAPE_UNPADDED[1],
        CHUNK_SHAPE_UNPADDED[2],
    )
    GLOBAL_INDEX_UNPADDED = (0, 0, 0)
    global_data = np.arange(np.prod(GLOBAL_SHAPE), dtype=np.float32).reshape(
        GLOBAL_SHAPE
    )
    aux_data = AuxiliaryData(angles=np.ones(GLOBAL_SHAPE[0], dtype=np.float32))

    # Define a mock loader (wrapper needs this for metadata, such as the detector shape) and a
    # mock "previous method" to the intermediate data wrapper
    loader: LoaderInterface = mocker.create_autospec(
        LoaderInterface, instance=True, detector_x=10, detector_y=20
    )
    prev_method = mocker.create_autospec(
        MethodWrapper,
        instance=True,
        task_id="task1",
        package_name="testpackage",
        method_name="testmethod",
        recon_algorithm="XXX",
    )

    # Define dummy method function that the intermediate data wrapper will be patched to import
    class FakeModule:
        def save_intermediate_data(
            data: np.ndarray,  # type: ignore
            global_shape: Tuple[int, int, int],
            global_index: Tuple[int, int, int],
            slicing_dim: int,
            file: h5py.File,
            frames_per_chunk: int,
            path: str,
            detector_x: int,
            detector_y: int,
            angles: np.ndarray,
        ):
            assert data.shape == BLOCK_SHAPE_UNPADDED
            assert global_index == GLOBAL_INDEX_UNPADDED

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )

    # Create intermediate data wrapper
    mocker.patch.object(httomo.globals, "run_out_dir", tmp_path)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="gpu_cupy"),
        "httomo.methods",
        "save_intermediate_data",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        loader=loader,
        prev_method=prev_method,
    )

    # Create block from the global/chunk data
    chunk_shape_list: List[int] = list(GLOBAL_SHAPE)
    if padding != (0, 0):
        chunk_shape_list[0] += padding[0] + padding[1]

    block_data = global_data[0 : GLOBAL_SHAPE[0] // 2, :, :]
    block_start = 0
    if padding != (0, 0):
        block_data = np.pad(block_data, pad_width=(padding, (0, 0), (0, 0)))
        block_start -= padding[0]

    block = DataSetBlock(
        data=block_data,
        aux_data=aux_data,
        block_start=block_start,
        chunk_start=0 if padding == (0, 0) else -padding[0],
        global_shape=GLOBAL_SHAPE,
        chunk_shape=make_3d_shape_from_shape(chunk_shape_list),
        padding=padding,
    )

    # Execute the padded block with the intermediate wrapper and let the assertions in the
    # dummy method function run the appropriate checks
    wrp.execute(block)


@pytest.mark.parametrize("recon_filename_stem_global_var", [None, "some-recon"])
@pytest.mark.parametrize(
    "recon_algorithm",
    [None, "gridrec"],
    ids=["specify-recon-algorithm", "dont-specify-recon-algorithm"],
)
def test_recon_method_output_filename(
    get_files: Callable,
    mocker: MockerFixture,
    tmp_path: Path,
    recon_filename_stem_global_var: Optional[str],
    recon_algorithm: Optional[str],
):
    httomo.globals.RECON_FILENAME_STEM = recon_filename_stem_global_var
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
            frames_per_chunk: int,
            path: str,
            detector_x: int,
            detector_y: int,
            angles: np.ndarray,
        ):
            pass

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    TASK_ID = "task1"
    PACKAGE_NAME = "testpackage"
    METHOD_NAME = "testreconmethod"
    MODULE_PATH = f"{PACKAGE_NAME}.algorithm"
    if recon_filename_stem_global_var is None and recon_algorithm is None:
        expected_filename = f"{TASK_ID}-{PACKAGE_NAME}-{METHOD_NAME}"
    if recon_filename_stem_global_var is None and recon_algorithm is not None:
        expected_filename = f"{TASK_ID}-{PACKAGE_NAME}-{METHOD_NAME}-{recon_algorithm}"
    if recon_filename_stem_global_var is not None:
        expected_filename = recon_filename_stem_global_var
    expected_filename += ".h5"
    prev_method = mocker.create_autospec(
        MethodWrapper,
        instance=True,
        task_id=TASK_ID,
        package_name=PACKAGE_NAME,
        method_name=METHOD_NAME,
        module_path=MODULE_PATH,
        recon_algorithm=recon_algorithm,
    )
    mocker.patch.object(httomo.globals, "run_out_dir", tmp_path)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="cpu"),
        "httomo.methods",
        "save_intermediate_data",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        loader=loader,
        prev_method=prev_method,
    )

    assert isinstance(wrp, SaveIntermediateFilesWrapper)
    files = get_files(tmp_path)
    assert len(files) == 1
    assert Path(files[0]).name == expected_filename


@pytest.mark.parametrize("recon_filename_stem_global_var", [None, "some-recon"])
def test_non_recon_method_output_filename(
    get_files: Callable,
    mocker: MockerFixture,
    tmp_path: Path,
    recon_filename_stem_global_var: Optional[str],
):
    httomo.globals.RECON_FILENAME_STEM = recon_filename_stem_global_var
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
            frames_per_chunk: int,
            path: str,
            detector_x: int,
            detector_y: int,
            angles: np.ndarray,
        ):
            pass

    mocker.patch(
        "httomo.method_wrappers.generic.import_module", return_value=FakeModule
    )
    TASK_ID = "task1"
    PACKAGE_NAME = "testpackage"
    METHOD_NAME = "testmethod"
    MODULE_PATH = f"{PACKAGE_NAME}.notalgorithm"
    EXPECTED_FILENAME = f"{TASK_ID}-{PACKAGE_NAME}-{METHOD_NAME}.h5"
    prev_method = mocker.create_autospec(
        MethodWrapper,
        instance=True,
        task_id=TASK_ID,
        package_name=PACKAGE_NAME,
        method_name=METHOD_NAME,
        module_path=MODULE_PATH,
        recon_algorithm=None,
    )
    mocker.patch.object(httomo.globals, "run_out_dir", tmp_path)
    wrp = make_method_wrapper(
        make_mock_repo(mocker, implementation="cpu"),
        "httomo.methods",
        "save_intermediate_data",
        MPI.COMM_WORLD,
        make_mock_preview_config(mocker),
        loader=loader,
        prev_method=prev_method,
    )

    assert isinstance(wrp, SaveIntermediateFilesWrapper)
    files = get_files(tmp_path)
    assert len(files) == 1
    assert Path(files[0]).name == EXPECTED_FILENAME
