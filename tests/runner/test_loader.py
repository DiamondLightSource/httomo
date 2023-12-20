import os
from pathlib import Path
from typing import Union

import h5py
from mpi4py import MPI
import pytest
from pytest_mock import MockerFixture
import numpy as np

from httomo.data.hdf.loaders import LoaderData
from httomo.runner.dataset import DataSet
from httomo.runner.loader import DarksFlatsFileConfig, RawAngles, StandardTomoLoader, UserDefinedAngles, get_darks_flats, make_loader
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
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    assert loader.slicing_dim == SLICING_DIM


def test_standard_tomo_loader_get_global_shape(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    GLOBAL_DATA_SHAPE = (180, 128, 160)
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    assert loader.global_shape == GLOBAL_DATA_SHAPE


def test_standard_tomo_loader_get_chunk_index_single_proc(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    CHUNK_INDEX = (0, 0, 0)
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    assert loader.chunk_index == CHUNK_INDEX


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_standard_tomo_loader_get_chunk_index_two_procs(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    GLOBAL_DATA_SHAPE = (180, 128, 160)

    chunk_index = (0, 0, 0) if COMM.rank == 0 else (GLOBAL_DATA_SHAPE[0] // 2, 0, 0)
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    assert loader.chunk_index == chunk_index


def test_standard_tomo_loader_get_chunk_shape_single_proc(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    CHUNK_SHAPE = (180, 128, 160)
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    assert loader.chunk_shape == CHUNK_SHAPE


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_standard_tomo_loader_get_chunk_shape_two_procs(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    GLOBAL_DATA_SHAPE = (180, 128, 160)
    CHUNK_SHAPE = (
        GLOBAL_DATA_SHAPE[0] // 2,
        GLOBAL_DATA_SHAPE[1],
        GLOBAL_DATA_SHAPE[2]
    )
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    assert loader.chunk_shape == CHUNK_SHAPE


def test_standard_tomo_loader_read_block_single_proc(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )

    BLOCK_START = 2
    BLOCK_LENGTH = 4
    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    PROJS_START = 0
    with h5py.File(standard_data, "r") as f:
        dataset: h5py.Dataset = f[standard_data_path]
        projs: np.ndarray = dataset[
            PROJS_START + BLOCK_START: PROJS_START + BLOCK_START + BLOCK_LENGTH
        ]

    assert projs.shape[SLICING_DIM] == BLOCK_LENGTH
    np.testing.assert_array_equal(block.data, projs)


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_standard_tomo_loader_read_block_two_procs(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )

    BLOCK_START = 2
    BLOCK_LENGTH = 4
    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    GLOBAL_DATA_SHAPE = (180, 128, 160)
    CHUNK_SHAPE = (
        GLOBAL_DATA_SHAPE[0] // 2,
        GLOBAL_DATA_SHAPE[1],
        GLOBAL_DATA_SHAPE[2]
    )

    projs_start = 0 if COMM.rank == 0 else CHUNK_SHAPE[0]
    with h5py.File(standard_data, "r") as f:
        dataset: h5py.Dataset = f[standard_data_path]
        projs: np.ndarray = dataset[
            projs_start + BLOCK_START: projs_start + BLOCK_START + BLOCK_LENGTH
        ]

    assert projs.shape[SLICING_DIM] == BLOCK_LENGTH
    np.testing.assert_array_equal(block.data, projs)


def test_standard_tomo_loader_read_block_adjust_for_darks_flats_single_proc(
    diad_data: str,
):
    DATA_PATH = "/entry/imaging/data"
    IMAGE_KEY_PATH = "/entry/instrument/imaging/image_key"
    ANGLES_CONFIG = RawAngles(
        data_path="/entry/imaging_sum/gts_theta_value"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(diad_data),
        data_path=DATA_PATH,
        image_key_path=IMAGE_KEY_PATH,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(diad_data),
        data_path=DATA_PATH,
        image_key_path=IMAGE_KEY_PATH,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )

    BLOCK_START = 0
    BLOCK_LENGTH = 4
    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    # Darks/flats are at indices 0 to 99 (and 3101 to 3200), projection data starts at index
    # 100
    PROJS_START = 100
    with h5py.File(diad_data, "r") as f:
        dataset: h5py.Dataset = f[DATA_PATH]
        projs: np.ndarray = dataset[
            PROJS_START + BLOCK_START: PROJS_START + BLOCK_START + BLOCK_LENGTH
        ]

    assert projs.shape[SLICING_DIM] == BLOCK_LENGTH
    np.testing.assert_array_equal(block.data, projs)


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_standard_tomo_loader_read_block_adjust_for_darks_flats_two_procs(
    diad_data: str,
):
    DATA_PATH = "/entry/imaging/data"
    IMAGE_KEY_PATH = "/entry/instrument/imaging/image_key"
    ANGLES_CONFIG = RawAngles(
        data_path="/entry/imaging_sum/gts_theta_value"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(diad_data),
        data_path=DATA_PATH,
        image_key_path=IMAGE_KEY_PATH,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(diad_data),
        data_path=DATA_PATH,
        image_key_path=IMAGE_KEY_PATH,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )

    BLOCK_START = 2
    BLOCK_LENGTH = 4
    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    GLOBAL_DATA_SHAPE = (3000, 22, 26)
    CHUNK_SHAPE = (
        GLOBAL_DATA_SHAPE[0] // 2,
        GLOBAL_DATA_SHAPE[1],
        GLOBAL_DATA_SHAPE[2]
    )

    # Darks/flats are at indices 0 to 99 (and 3101 to 3200), projection data starts at index
    # 100
    PROJS_SHIFT = 100
    projs_start = PROJS_SHIFT if COMM.rank == 0 else PROJS_SHIFT + CHUNK_SHAPE[0]
    with h5py.File(diad_data, "r") as f:
        dataset: h5py.Dataset = f[DATA_PATH]
        projs: np.ndarray = dataset[
            projs_start + BLOCK_START: projs_start + BLOCK_START + BLOCK_LENGTH
        ]

    assert projs.shape[SLICING_DIM] == BLOCK_LENGTH
    np.testing.assert_array_equal(block.data, projs)


def test_standard_tomo_loader_generates_block_with_angles(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )

    BLOCK_START = 0
    BLOCK_LENGTH = 2
    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    with h5py.File(standard_data, "r") as f:
        dataset: h5py.Dataset = f[ANGLES_CONFIG.data_path]
        angles = dataset[...]

    np.testing.assert_array_equal(block.angles, angles)


def test_standard_tomo_loader_user_defined_angles(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    SLICING_DIM = 0
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    COMM = MPI.COMM_WORLD
    # Override angles in raw data with the config for some arbitrary array
    USER_DEFINED_ANGLES = UserDefinedAngles(
        start_angle=0,
        stop_angle=180,
        angles_total=720,
    )
    EXPECTED_ANGLES = np.linspace(
        USER_DEFINED_ANGLES.start_angle,
        USER_DEFINED_ANGLES.stop_angle,
        USER_DEFINED_ANGLES.angles_total,
    )
    loader = StandardTomoLoader(
        in_file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
        darks=DARKS_CONFIG,
        flats=FLATS_CONFIG,
        angles=USER_DEFINED_ANGLES,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )

    BLOCK_START = 0
    BLOCK_LENGTH = 2
    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)
    np.testing.assert_array_equal(block.angles, EXPECTED_ANGLES)


def test_get_darks_flats_same_file_same_dataset(
    standard_data: str,
    standard_data_path: str,
    standard_image_key_path: str,
):
    COMM = MPI.COMM_WORLD
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(standard_data),
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )

    loaded_darks, loaded_flats = get_darks_flats(
        DARKS_CONFIG,
        FLATS_CONFIG,
        COMM,
    )

    FLATS_START = 180
    FLATS_END = 199
    DARKS_START = 200
    DARKS_END = 219
    with h5py.File(standard_data, "r") as f:
        dataset: h5py.Dataset = f[standard_data_path]
        flats = dataset[FLATS_START:FLATS_END + 1]
        darks = dataset[DARKS_START:DARKS_END + 1]

    np.testing.assert_array_equal(loaded_flats, flats)
    np.testing.assert_array_equal(loaded_darks, darks)


def test_get_darks_flats_different_file():
    COMM = MPI.COMM_WORLD
    DARKS_CONFIG = DarksFlatsFileConfig(
        file=Path("tests/test_data/i12/separate_flats_darks/dark_field.h5"),
        data_path="/1-NoProcessPlugin-tomo/data",
        image_key_path=None,
    )
    FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path("tests/test_data/i12/separate_flats_darks/flat_field.h5"),
        data_path="/1-NoProcessPlugin-tomo/data",
        image_key_path=None,
    )

    loaded_darks, loaded_flats = get_darks_flats(
        DARKS_CONFIG,
        FLATS_CONFIG,
        COMM,
    )

    with h5py.File(DARKS_CONFIG.file, "r") as f:
        darks = f[DARKS_CONFIG.data_path][...]

    with h5py.File(FLATS_CONFIG.file, "r") as f:
        flats = f[FLATS_CONFIG.data_path][...]

    np.testing.assert_array_equal(loaded_flats, flats)
    np.testing.assert_array_equal(loaded_darks, darks)
