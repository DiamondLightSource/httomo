from pathlib import Path
from typing import Tuple

import pytest
import h5py
import numpy as np

from httomo.preview import Preview, PreviewConfig, PreviewDimConfig
from httomo.transform_loader_params import (
    PreviewParam,
    parse_preview,
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
            "Preview indices in detector_y dim exceed bounds of data: start=0, stop=129",
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=161),
            ),
            True,
            "Preview indices in detector_x dim exceed bounds of data: start=0, stop=161",
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=220, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            True,
            (
                "Preview index error for angles: start must be strictly smaller than "
                "stop, but start=220, stop=220"
            ),
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=60, stop=50),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
            True,
            (
                "Preview index error for detector_y: start must be strictly smaller than "
                "stop, but start=60, stop=50"
            ),
        ),
        (
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=50, stop=0),
            ),
            True,
            (
                "Preview index error for detector_x: start must be strictly smaller than "
                "stop, but start=50, stop=0"
            ),
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
        "start_geq_stop_det_angles_bounds",
        "start_geq_stop_det_y_bounds",
        "start_geq_stop_det_x_bounds",
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
    IN_FILE = Path(__file__).parent / "test_data/tomo_standard.nxs"
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
    standard_image_key_path: str,
):
    IN_FILE_PATH = Path(__file__).parent / "test_data/tomo_standard.nxs"
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
        Path(__file__).parent
        / "test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
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
    standard_image_key_path: str,
    previewed_shape: Tuple[int, int, int],
):
    IN_FILE_PATH = Path(__file__).parent / "test_data/tomo_standard.nxs"
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
