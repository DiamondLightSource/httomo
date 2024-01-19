import os
from pathlib import Path
from typing import Union
from unittest import mock

import h5py
from mpi4py import MPI
import pytest
from pytest_mock import MockerFixture
import numpy as np

from httomo.data.hdf.loaders import LoaderData
from httomo.runner.dataset import DataSet
from httomo.runner.loader import DarksFlatsFileConfig, RawAngles, StandardTomoLoader, UserDefinedAngles, get_darks_flats
from ..testing_utils import make_mock_repo


def make_standard_tomo_loader() -> StandardTomoLoader:
    """
    Create an instance of `StandardTomoLoader` with some commonly used default values for
    loading the test data `tomo_standard.nxs`.
    """
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    DARKS_FLATS_CONFIG = DarksFlatsFileConfig(
        file=IN_FILE_PATH,
        data_path="/entry1/tomo_entry/data/data",
        image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
    )
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD
    loader = StandardTomoLoader(
        in_file=IN_FILE_PATH,
        data_path=DARKS_FLATS_CONFIG.data_path,
        image_key_path=DARKS_FLATS_CONFIG.image_key_path,
        darks=DARKS_FLATS_CONFIG,
        flats=DARKS_FLATS_CONFIG,
        angles=ANGLES_CONFIG,
        slicing_dim=SLICING_DIM,
        comm=COMM,
    )
    return loader

def test_standard_tomo_loader_get_slicing_dim():
    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()
    assert loader.slicing_dim == 0


def test_standard_tomo_loader_get_global_shape():
    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()
    GLOBAL_DATA_SHAPE = (180, 128, 160)
    assert loader.global_shape == GLOBAL_DATA_SHAPE


def test_standard_tomo_loader_get_chunk_index_single_proc():
    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()
    CHUNK_INDEX = (0, 0, 0)
    assert loader.chunk_index == CHUNK_INDEX


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_standard_tomo_loader_get_chunk_index_two_procs():
    COMM = MPI.COMM_WORLD
    GLOBAL_DATA_SHAPE = (180, 128, 160)
    chunk_index = (0, 0, 0) if COMM.rank == 0 else (GLOBAL_DATA_SHAPE[0] // 2, 0, 0)
    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()
    assert loader.chunk_index == chunk_index


def test_standard_tomo_loader_get_chunk_shape_single_proc():
    CHUNK_SHAPE = (180, 128, 160)
    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()
    assert loader.chunk_shape == CHUNK_SHAPE


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
def test_standard_tomo_loader_get_chunk_shape_two_procs():
    GLOBAL_DATA_SHAPE = (180, 128, 160)
    CHUNK_SHAPE = (
        GLOBAL_DATA_SHAPE[0] // 2,
        GLOBAL_DATA_SHAPE[1],
        GLOBAL_DATA_SHAPE[2]
    )
    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()
    assert loader.chunk_shape == CHUNK_SHAPE


def test_standard_tomo_loader_read_block_single_proc(
    standard_data_path: str,
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    SLICING_DIM = 0
    BLOCK_START = 2
    BLOCK_LENGTH = 4
    PROJS_START = 0

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()

    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    with h5py.File(IN_FILE_PATH, "r") as f:
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
    standard_data_path: str,
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD

    BLOCK_START = 2
    BLOCK_LENGTH = 4

    GLOBAL_DATA_SHAPE = (180, 128, 160)
    CHUNK_SHAPE = (
        GLOBAL_DATA_SHAPE[0] // 2,
        GLOBAL_DATA_SHAPE[1],
        GLOBAL_DATA_SHAPE[2]
    )

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()

    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    projs_start = 0 if COMM.rank == 0 else CHUNK_SHAPE[0]
    with h5py.File(IN_FILE_PATH, "r") as f:
        dataset: h5py.Dataset = f[standard_data_path]
        projs: np.ndarray = dataset[
            projs_start + BLOCK_START: projs_start + BLOCK_START + BLOCK_LENGTH
        ]

    assert projs.shape[SLICING_DIM] == BLOCK_LENGTH
    np.testing.assert_array_equal(block.data, projs)


def test_standard_tomo_loader_read_block_adjust_for_darks_flats_single_proc():
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/k11_diad/k11-18014.nxs"
    DATA_PATH = "/entry/imaging/data"
    IMAGE_KEY_PATH = "/entry/instrument/imaging/image_key"
    ANGLES_CONFIG = RawAngles(
        data_path="/entry/imaging_sum/gts_theta_value"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=IN_FILE_PATH,
        data_path=DATA_PATH,
        image_key_path=IMAGE_KEY_PATH,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = StandardTomoLoader(
            in_file=IN_FILE_PATH,
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
    with h5py.File(IN_FILE_PATH, "r") as f:
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
def test_standard_tomo_loader_read_block_adjust_for_darks_flats_two_procs():
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/k11_diad/k11-18014.nxs"
    DATA_PATH = "/entry/imaging/data"
    IMAGE_KEY_PATH = "/entry/instrument/imaging/image_key"
    ANGLES_CONFIG = RawAngles(
        data_path="/entry/imaging_sum/gts_theta_value"
    )
    DARKS_CONFIG = FLATS_CONFIG = DarksFlatsFileConfig(
        file=IN_FILE_PATH,
        data_path=DATA_PATH,
        image_key_path=IMAGE_KEY_PATH,
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = StandardTomoLoader(
            in_file=IN_FILE_PATH,
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
    with h5py.File(IN_FILE_PATH, "r") as f:
        dataset: h5py.Dataset = f[DATA_PATH]
        projs: np.ndarray = dataset[
            projs_start + BLOCK_START: projs_start + BLOCK_START + BLOCK_LENGTH
        ]

    assert projs.shape[SLICING_DIM] == BLOCK_LENGTH
    np.testing.assert_array_equal(block.data, projs)


def test_standard_tomo_loader_generates_block_with_angles():
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    ANGLES_PATH = "/entry1/tomo_entry/data/rotation_angle"
    BLOCK_START = 0
    BLOCK_LENGTH = 2

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = make_standard_tomo_loader()

    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    with h5py.File(IN_FILE_PATH, "r") as f:
        dataset: h5py.Dataset = f[ANGLES_PATH]
        angles = dataset[...]

    np.testing.assert_array_equal(block.angles, angles)


def test_standard_tomo_loader_user_defined_angles(
    standard_data_path: str,
    standard_image_key_path: str,
    standard_data_darks_flats_config: DarksFlatsFileConfig,
):
    SLICING_DIM = 0
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

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = StandardTomoLoader(
            in_file=Path(__file__).parent.parent / "test_data/tomo_standard.nxs",
            data_path=standard_data_path,
            image_key_path=standard_image_key_path,
            darks=standard_data_darks_flats_config,
            flats=standard_data_darks_flats_config,
            angles=USER_DEFINED_ANGLES,
            slicing_dim=SLICING_DIM,
            comm=COMM,
        )

    BLOCK_START = 0
    BLOCK_LENGTH = 2
    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)
    np.testing.assert_array_equal(block.angles, EXPECTED_ANGLES)



def test_standard_tomo_loader_closes_file(mocker: MockerFixture):
    loader = make_standard_tomo_loader()
    file_close = mocker.patch.object(loader._h5file, "close")
    loader.finalize()
    file_close.assert_called_once()


def test_get_darks_flats_same_file_same_dataset(
    standard_data_path: str,
    standard_data_darks_flats_config: DarksFlatsFileConfig,
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    COMM = MPI.COMM_WORLD

    loaded_darks, loaded_flats = get_darks_flats(
        standard_data_darks_flats_config,
        standard_data_darks_flats_config,
        COMM,
    )

    FLATS_START = 180
    FLATS_END = 199
    DARKS_START = 200
    DARKS_END = 219
    with h5py.File(IN_FILE_PATH, "r") as f:
        dataset: h5py.Dataset = f[standard_data_path]
        flats = dataset[FLATS_START:FLATS_END + 1]
        darks = dataset[DARKS_START:DARKS_END + 1]

    np.testing.assert_array_equal(loaded_flats, flats)
    np.testing.assert_array_equal(loaded_darks, darks)


def test_get_darks_flats_different_file():
    COMM = MPI.COMM_WORLD
    DARKS_CONFIG = DarksFlatsFileConfig(
        file=Path(__file__).parent.parent / "test_data/i12/separate_flats_darks/dark_field.h5",
        data_path="/1-NoProcessPlugin-tomo/data",
        image_key_path=None,
    )
    FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(__file__).parent.parent / "test_data/i12/separate_flats_darks/flat_field.h5",
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
