from pathlib import Path

import pytest
import h5py
import numpy as np

from httomo.darks_flats import DarksFlatsFileConfig, get_darks_flats
from httomo.preview import PreviewConfig, PreviewDimConfig


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
def test_get_darks_flats_same_file_same_dataset(
    standard_data_path: str,
    standard_data_darks_flats_config: DarksFlatsFileConfig,
    preview_config: PreviewConfig,
):
    IN_FILE_PATH = Path(__file__).parent / "test_data/tomo_standard.nxs"

    loaded_darks, loaded_flats = get_darks_flats(
        standard_data_darks_flats_config,
        standard_data_darks_flats_config,
        preview_config,
    )

    FLATS_START = 180
    FLATS_END = 199
    DARKS_START = 200
    DARKS_END = 219
    with h5py.File(IN_FILE_PATH, "r") as f:
        dataset: h5py.Dataset = f[standard_data_path]
        flats = dataset[
            FLATS_START : FLATS_END + 1,
            preview_config.detector_y.start : preview_config.detector_y.stop,
            preview_config.detector_x.start : preview_config.detector_x.stop,
        ]
        darks = dataset[
            DARKS_START : DARKS_END + 1,
            preview_config.detector_y.start : preview_config.detector_y.stop,
            preview_config.detector_x.start : preview_config.detector_x.stop,
        ]

    np.testing.assert_array_equal(loaded_flats, flats)
    np.testing.assert_array_equal(loaded_darks, darks)


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
def test_get_darks_flats_different_file(preview_config: PreviewConfig):
    DARKS_CONFIG = DarksFlatsFileConfig(
        file=Path(__file__).parent / "test_data/i12/separate_flats_darks/dark_field.h5",
        data_path="/1-NoProcessPlugin-tomo/data",
        image_key_path=None,
    )
    FLATS_CONFIG = DarksFlatsFileConfig(
        file=Path(__file__).parent / "test_data/i12/separate_flats_darks/flat_field.h5",
        data_path="/1-NoProcessPlugin-tomo/data",
        image_key_path=None,
    )

    loaded_darks, loaded_flats = get_darks_flats(
        DARKS_CONFIG,
        FLATS_CONFIG,
        preview_config,
    )

    with h5py.File(DARKS_CONFIG.file, "r") as f:
        darks = f[DARKS_CONFIG.data_path][
            :,
            preview_config.detector_y.start : preview_config.detector_y.stop,
            preview_config.detector_x.start : preview_config.detector_x.stop,
        ]

    with h5py.File(FLATS_CONFIG.file, "r") as f:
        flats = f[FLATS_CONFIG.data_path][
            :,
            preview_config.detector_y.start : preview_config.detector_y.stop,
            preview_config.detector_x.start : preview_config.detector_x.stop,
        ]

    np.testing.assert_array_equal(loaded_flats, flats)
    np.testing.assert_array_equal(loaded_darks, darks)
