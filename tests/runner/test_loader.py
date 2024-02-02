import os
from pathlib import Path
from typing import Tuple, Union
from unittest import mock

import h5py
from mpi4py import MPI
import pytest
from pytest_mock import MockerFixture
import numpy as np

from httomo.data.hdf.loaders import LoaderData
from httomo.runner.dataset import DataSet
from httomo.runner.loader import DarksFlatsFileConfig, Preview, PreviewConfig, PreviewDimConfig, RawAngles, StandardTomoLoader, UserDefinedAngles, get_darks_flats
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
    PREVIEW_CONFIG = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=180),
        detector_y=PreviewDimConfig(start=0, stop=128),
        detector_x=PreviewDimConfig(start=0, stop=160),
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
        preview_config=PREVIEW_CONFIG,
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


@pytest.mark.parametrize(
    "preview_config, expected_chunk_shape",
    [
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=180),
                detector_y=PreviewDimConfig(start=5, stop=15),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            (180, 10, 160),
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=180),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=5, stop=15),
            ),
            (180, 128, 10),
        ),
    ],
    ids=["crop_det_y", "crop_det_x"],
)
def test_standard_tomo_loader_previewed_get_chunk_shape_single_proc(
    standard_data_path: str,
    standard_image_key_path: str,
    preview_config: PreviewConfig,
    expected_chunk_shape: Tuple[int, int, int],
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    DARKS_FLATS_CONFIG = DarksFlatsFileConfig(
        file=IN_FILE_PATH,
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = StandardTomoLoader(
            in_file=IN_FILE_PATH,
            data_path=DARKS_FLATS_CONFIG.data_path,
            image_key_path=DARKS_FLATS_CONFIG.image_key_path,
            darks=DARKS_FLATS_CONFIG,
            flats=DARKS_FLATS_CONFIG,
            angles=ANGLES_CONFIG,
            preview_config=preview_config,
            slicing_dim=SLICING_DIM,
            comm=COMM,
        )

    assert loader.chunk_shape == expected_chunk_shape


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size != 2, reason="Only rank-2 MPI is supported with this test"
)
@pytest.mark.parametrize(
    "preview_config, expected_chunk_shape",
    [
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=180),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            (90, 128, 160),
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=180),
                detector_y=PreviewDimConfig(start=5, stop=15),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            (90, 10, 160),
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=180),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=5, stop=15),
            ),
            (90, 128, 10),
        ),
    ],
    ids=["no_cropping", "crop_det_y", "crop_det_x"],
)
def test_standard_tomo_loader_get_chunk_shape_two_procs(
    standard_data_path: str,
    standard_image_key_path: str,
    preview_config: PreviewConfig,
    expected_chunk_shape: Tuple[int, int, int],
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    DARKS_FLATS_CONFIG = DarksFlatsFileConfig(
        file=IN_FILE_PATH,
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = StandardTomoLoader(
            in_file=IN_FILE_PATH,
            data_path=DARKS_FLATS_CONFIG.data_path,
            image_key_path=DARKS_FLATS_CONFIG.image_key_path,
            darks=DARKS_FLATS_CONFIG,
            flats=DARKS_FLATS_CONFIG,
            angles=ANGLES_CONFIG,
            preview_config=preview_config,
            slicing_dim=SLICING_DIM,
            comm=COMM,
        )

    assert loader.chunk_shape == expected_chunk_shape


@pytest.mark.parametrize(
    "preview_config",
    [
        PreviewConfig(
            angles=PreviewDimConfig(start=0, stop=180),
            detector_y=PreviewDimConfig(start=0, stop=128),
            detector_x=PreviewDimConfig(start=0, stop=160),
        ),
        PreviewConfig(
            angles=PreviewDimConfig(start=0, stop=180),
            detector_y=PreviewDimConfig(start=0, stop=10),
            detector_x=PreviewDimConfig(start=0, stop=160),
        ),
        PreviewConfig(
            angles=PreviewDimConfig(start=0, stop=180),
            detector_y=PreviewDimConfig(start=10, stop=20),
            detector_x=PreviewDimConfig(start=0, stop=160),
        ),
        PreviewConfig(
            angles=PreviewDimConfig(start=0, stop=180),
            detector_y=PreviewDimConfig(start=0, stop=128),
            detector_x=PreviewDimConfig(start=0, stop=10),
        ),
        PreviewConfig(
            angles=PreviewDimConfig(start=0, stop=180),
            detector_y=PreviewDimConfig(start=0, stop=128),
            detector_x=PreviewDimConfig(start=10, stop=20),
        ),
    ],
    ids=[
        "no_cropping",
        "crop_det_y_start_0",
        "crop_det_y_start_10",
        "crop_det_x_start_0",
        "crop_det_x_start_10",
    ],
)
def test_standard_tomo_loader_read_block_single_proc(
    standard_data_path: str,
    standard_image_key_path: str,
    preview_config: PreviewConfig,
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    DARKS_FLATS_CONFIG = DarksFlatsFileConfig(
        file=IN_FILE_PATH,
        data_path=standard_data_path,
        image_key_path=standard_image_key_path,
    )
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    SLICING_DIM = 0
    COMM = MPI.COMM_WORLD

    BLOCK_START = 2
    BLOCK_LENGTH = 4
    PROJS_START = 0
    expected_block_shape = (
        BLOCK_LENGTH,
        preview_config.detector_y.stop - preview_config.detector_y.start,
        preview_config.detector_x.stop - preview_config.detector_x.start,
    )

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ):
        loader = StandardTomoLoader(
            in_file=IN_FILE_PATH,
            data_path=DARKS_FLATS_CONFIG.data_path,
            image_key_path=DARKS_FLATS_CONFIG.image_key_path,
            darks=DARKS_FLATS_CONFIG,
            flats=DARKS_FLATS_CONFIG,
            angles=ANGLES_CONFIG,
            preview_config=preview_config,
            slicing_dim=SLICING_DIM,
            comm=COMM,
        )

    block = loader.read_block(BLOCK_START, BLOCK_LENGTH)

    with h5py.File(IN_FILE_PATH, "r") as f:
        dataset: h5py.Dataset = f[standard_data_path]
        projs: np.ndarray = dataset[
            PROJS_START + BLOCK_START: PROJS_START + BLOCK_START + BLOCK_LENGTH,
            preview_config.detector_y.start : preview_config.detector_y.stop,
            preview_config.detector_x.start : preview_config.detector_x.stop,
        ]

    assert block.data.shape == expected_block_shape
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

    assert block.data.shape[SLICING_DIM] == BLOCK_LENGTH
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
    PREVIEW_CONFIG = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=3201),
        detector_y=PreviewDimConfig(start=0, stop=22),
        detector_x=PreviewDimConfig(start=0, stop=26),
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
            preview_config=PREVIEW_CONFIG,
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

    assert block.data.shape[SLICING_DIM] == BLOCK_LENGTH
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
    PREVIEW_CONFIG = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=3201),
        detector_y=PreviewDimConfig(start=0, stop=22),
        detector_x=PreviewDimConfig(start=0, stop=26),
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
            preview_config=PREVIEW_CONFIG,
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

    assert block.data.shape[SLICING_DIM] == BLOCK_LENGTH
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
    PREVIEW_CONFIG = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=180),
        detector_y=PreviewDimConfig(start=0, stop=128),
        detector_x=PreviewDimConfig(start=0, stop=160),
    )
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
            preview_config=PREVIEW_CONFIG,
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


def test_standard_tomo_loader_raises_error_slicing_dim(
    standard_data_darks_flats_config: DarksFlatsFileConfig,
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    ANGLES_CONFIG = RawAngles(
        data_path="/entry1/tomo_entry/data/rotation_angle"
    )
    PREVIEW_CONFIG = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=180),
        detector_y=PreviewDimConfig(start=0, stop=128),
        detector_x=PreviewDimConfig(start=0, stop=160),
    )
    SLICING_DIM = 1
    COMM = MPI.COMM_WORLD

    with mock.patch(
        "httomo.runner.loader.get_darks_flats",
        return_value=(np.zeros(1), np.zeros(1)),
    ), pytest.raises(NotImplementedError):
        _ = StandardTomoLoader(
            in_file=IN_FILE_PATH,
            data_path=standard_data_darks_flats_config.data_path,
            image_key_path=standard_data_darks_flats_config.image_key_path,
            darks=standard_data_darks_flats_config,
            flats=standard_data_darks_flats_config,
            preview_config=PREVIEW_CONFIG,
            angles=ANGLES_CONFIG,
            slicing_dim=SLICING_DIM,
            comm=COMM,
        )


@pytest.mark.parametrize(
    "preview_config, is_error_expected, err_str",
    [
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=221),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            True,
            "Preview indices in angles dim exceed bounds of data: start=0, stop=221",
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=129),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            True,
            "Preview indices in det y dim exceed bounds of data: start=0, stop=129",
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=161),
            ),
            True,
            "Preview indices in det x dim exceed bounds of data: start=0, stop=161",
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            False,
            "",
        ),
    ],
    ids=[
        "incorrect_angles_bounds",
        "incorrect_det_y_bounds",
        "incorrect_det_x_bounds",
        "all_correct_bounds",
    ],
)
def test_preview_bound_checking(
    standard_data_path: str,
    standard_image_key_path: str,
    preview_config: PreviewConfig,
    is_error_expected: bool,
    err_str: str,
):
    IN_FILE = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    f = h5py.File(IN_FILE, "r")
    dataset = f[standard_data_path]
    image_key = f[standard_image_key_path]

    if is_error_expected:
        with pytest.raises(ValueError, match=err_str):
            _ = Preview(
                preview_config=preview_config,
                dataset=dataset,
                image_key=image_key,
            )
    else:
        preview = Preview(
            preview_config=preview_config,
            dataset=dataset,
            image_key=image_key,
        )
        assert preview.config == preview_config

    f.close()


def test_preview_calculate_data_indices_excludes_darks_flats(
    standard_data_path: str,
    standard_image_key_path:str,
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    f = h5py.File(IN_FILE_PATH, "r")
    dataset = f[standard_data_path]
    image_key = f[standard_image_key_path]
    all_indices: np.ndarray = image_key[:]
    data_indices = np.where(all_indices == 0)[0]

    config = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=220),
        detector_y=PreviewDimConfig(start=0, stop=128),
        detector_x=PreviewDimConfig(start=0, stop=160),
    )
    preview = Preview(
        preview_config=config,
        dataset=dataset,
        image_key=image_key,
    )
    assert not np.array_equal(preview.data_indices, all_indices)
    assert np.array_equal(preview.data_indices, data_indices)
    assert preview.config.angles == PreviewDimConfig(start=0, stop=180)
    f.close()


def test_preview_with_no_image_key():
    IN_FILE_PATH = (
        Path(__file__).parent.parent /
            "test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
    )
    f = h5py.File(IN_FILE_PATH, "r")
    DATA_PATH = "1-TempPlugin-tomo/data"
    ANGLES, DET_Y, DET_X = (724, 10, 192)
    expected_indices = list(range(ANGLES))
    config = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=ANGLES),
        detector_y=PreviewDimConfig(start=0, stop=DET_Y),
        detector_x=PreviewDimConfig(start=0, stop=DET_X),
    )
    preview = Preview(
        preview_config=config,
        dataset=f[DATA_PATH],
        image_key=None,
    )
    assert np.array_equal(preview.data_indices, expected_indices)


@pytest.mark.parametrize(
    "previewed_shape",
    [(100, 128, 160), (180, 10, 160), (180, 128, 10)],
    ids=["crop_angles_dim", "crop_det_y_dim", "crop_det_x_dim"],
)
def test_preview_global_shape(
    standard_data_path: str,
    standard_image_key_path:str,
    previewed_shape: Tuple[int, int, int],
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"
    f = h5py.File(IN_FILE_PATH, "r")
    dataset = f[standard_data_path]
    image_key = f[standard_image_key_path]

    config = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=previewed_shape[0]),
        detector_y=PreviewDimConfig(start=0, stop=previewed_shape[1]),
        detector_x=PreviewDimConfig(start=0, stop=previewed_shape[2]),
    )
    preview = Preview(
        preview_config=config,
        dataset=dataset,
        image_key=image_key,
    )
    assert preview.global_shape == previewed_shape


def test_get_darks_flats_same_file_same_dataset(
    standard_data_path: str,
    standard_data_darks_flats_config: DarksFlatsFileConfig,
):
    IN_FILE_PATH = Path(__file__).parent.parent / "test_data/tomo_standard.nxs"

    loaded_darks, loaded_flats = get_darks_flats(
        standard_data_darks_flats_config,
        standard_data_darks_flats_config,
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
    )

    with h5py.File(DARKS_CONFIG.file, "r") as f:
        darks = f[DARKS_CONFIG.data_path][...]

    with h5py.File(FLATS_CONFIG.file, "r") as f:
        flats = f[FLATS_CONFIG.data_path][...]

    np.testing.assert_array_equal(loaded_flats, flats)
    np.testing.assert_array_equal(loaded_darks, darks)
