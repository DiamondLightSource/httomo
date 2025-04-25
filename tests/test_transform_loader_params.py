from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from httomo.darks_flats import DarksFlatsFileConfig
from httomo.loaders.types import (
    AnglesConfig,
    DataConfig,
    RawAngles,
    UserDefinedAngles,
)
from httomo.preview import PreviewConfig, PreviewDimConfig
from httomo.transform_loader_params import (
    DarksFlatsParam,
    PreviewParam,
    find_tomo_entry,
    parse_angles,
    parse_config,
    parse_darks_flats,
    parse_data,
    parse_preview,
)


@pytest.mark.parametrize(
    "data_shape, preview_param_value, expected_preview_config",
    [
        (
            (220, 128, 160),
            None,
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": {"start": 0, "stop": 128},
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {"angles": None, "detector_y": None, "detector_x": None},
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": None,
                "detector_y": {"start": 0, "stop": 128},
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": None,
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": {"start": 0, "stop": 128},
                "detector_x": None,
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "detector_y": {"start": 0, "stop": 128},
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": {"start": 0, "stop": 128},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": "mid",
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=62, stop=65),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 127, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": "mid",
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=62, stop=64),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 3, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": "mid",
                "detector_x": None,
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=3),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": {"stop": 128},
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            {
                "angles": {"start": 0, "stop": 220},
                "detector_y": {"start": 0},
                "detector_x": {"start": 0, "stop": 160},
            },
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
    ],
    ids=[
        "preview_param_none",
        "all_dims_provided",
        "all_dims_none",
        "angles_none",
        "det_y_none",
        "det_x_none",
        "missing_angles",
        "missing_det_y",
        "missing_det_x",
        "det_y_even_len_get_mid",
        "det_y_odd_len_get_mid",
        "det_y_small_len_get_mid",
        "det_y_missing_start",
        "det_y_missing_stop",
    ],
)
def test_parse_preview(
    data_shape: tuple[int, int, int],
    preview_param_value: PreviewParam,
    expected_preview_config: PreviewConfig,
):
    parsed_preview_config = parse_preview(preview_param_value, data_shape)
    assert parsed_preview_config == expected_preview_config


def test_parse_preview_raises_error_mid_in_angle_dim():
    DATA_SHAPE = (220, 128, 160)
    PREVIEW_PARAM_VALUE = {
        "angles": "mid",
        "detector_y": {"start": 0, "stop": 128},
        "detector_x": {"start": 0, "stop": 160},
    }
    with pytest.raises(ValueError):
        _ = parse_preview(PREVIEW_PARAM_VALUE, DATA_SHAPE)


@pytest.mark.parametrize(
    "angles_param, expected_angles_config",
    [
        (
            {"data_path": "/entry1/tomo_entry/data/rotation_angle"},
            RawAngles(data_path="/entry1/tomo_entry/data/rotation_angle"),
        ),
        (
            {
                "user_defined": {
                    "start_angle": 0,
                    "stop_angle": 180,
                    "angles_total": 724,
                },
            },
            UserDefinedAngles(
                start_angle=0,
                stop_angle=180,
                angles_total=724,
            ),
        ),
    ],
    ids=["raw_angles", "user_defined_angles"],
)
def test_parse_angles(angles_param: dict, expected_angles_config: AnglesConfig):
    angles_config = parse_angles(angles_param)
    assert angles_config == expected_angles_config


def test_preview_offset():
    data_shape = (220, 128, 160)

    preview_config_expected = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=data_shape[0]),
        detector_y=PreviewDimConfig(start=0, stop=data_shape[1]),
        detector_x=PreviewDimConfig(start=10, stop=data_shape[2] - 10),
    )

    param_value = PreviewParam(
        angles=None,
        detector_y={"start": 20, "start_offset": -21, "stop": 120, "stop_offset": 21},
        detector_x={
            "start": 0,
            "start_offset": 10,
            "stop": data_shape[2],
            "stop_offset": -10,
        },
    )
    preview_config = parse_preview(param_value=param_value, data_shape=data_shape)

    assert preview_config == preview_config_expected


def test_preview_keywords_offset():
    data_shape = (220, 128, 160)

    preview_config_expected = PreviewConfig(
        angles=PreviewDimConfig(start=0, stop=data_shape[0]),
        detector_y=PreviewDimConfig(start=10, stop=data_shape[1] - 10),
        detector_x=PreviewDimConfig(
            start=data_shape[2] // 2 - 50 - 1, stop=data_shape[2] // 2 + 50 - 1
        ),
    )

    param_value = PreviewParam(
        angles=None,
        detector_y={
            "start": "begin",
            "start_offset": 10,
            "stop": "end",
            "stop_offset": -10,
        },
        detector_x={
            "start": "mid",
            "start_offset": -50,
            "stop": "mid",
            "stop_offset": 50,
        },
    )
    preview_config = parse_preview(param_value=param_value, data_shape=data_shape)

    assert preview_config == preview_config_expected


def test_parse_preview_raises_error_if_stop_smaller_start():
    param_value = PreviewParam(
        angles=None,
        detector_y={
            "start": "begin",
            "start_offset": 10,
            "stop": "begin",
            "stop_offset": 5,
        },
        detector_x={
            "start": "mid",
            "start_offset": -50,
            "stop": "mid",
            "stop_offset": 50,
        },
    )

    with pytest.raises(ValueError) as e:
        _ = parse_preview(param_value=param_value, data_shape=(220, 128, 160))

    assert (
        "Stop value 5 is smaller or equal compared to the start value 10. Please check your preview values."
        in str(e)
    )


@pytest.mark.parametrize(
    "input",
    ["begin", "mid", "end", None, 10, "bla"],
)
def test_keywords_converter(input):
    data_shape1 = (220, 128, 160)
    if input == "begin":
        param_value = PreviewParam(
            angles=None,
            detector_y={
                "start": input,
                "stop": 10,
            },
            detector_x=None,
        )
        preview_config = parse_preview(param_value=param_value, data_shape=data_shape1)
        assert preview_config.detector_y.start == 0
    elif input == "mid":
        param_value = PreviewParam(
            angles=None,
            detector_y={
                "start": input,
                "stop": "end",
            },
            detector_x=None,
        )
        preview_config = parse_preview(param_value=param_value, data_shape=data_shape1)
        assert preview_config.detector_y.start == data_shape1[1] // 2 - 1
    elif input == "end":
        param_value = PreviewParam(
            angles=None,
            detector_y={
                "start": 0,
                "stop": input,
            },
            detector_x=None,
        )
        preview_config = parse_preview(param_value=param_value, data_shape=data_shape1)
        assert preview_config.detector_y.stop == data_shape1[1]
    elif input == None:
        param_value = PreviewParam(
            angles=None,
            detector_y=None,
            detector_x=None,
        )
        preview_config = parse_preview(param_value=param_value, data_shape=data_shape1)
        assert preview_config.detector_y.start == 0
        assert preview_config.detector_y.stop == data_shape1[1]
    elif input == 10:
        param_value = PreviewParam(
            angles=None,
            detector_y={
                "start": 0,
                "stop": input,
            },
            detector_x=None,
        )
        preview_config = parse_preview(param_value=param_value, data_shape=data_shape1)
        assert preview_config.detector_y.stop == 10
    else:
        param_value = PreviewParam(
            angles=None,
            detector_y={
                "start": 0,
                "stop": input,
            },
            detector_x=None,
        )
        with pytest.raises(ValueError) as e:
            _ = preview_config = parse_preview(
                param_value=param_value, data_shape=data_shape1
            )
            assert (
                "The given keyword: bla is not recognised. The recognised keywords are: begin, mid, end."
                in str(e)
            )


def test_parse_data():
    INPUT_FILE = "/some/path/to/data.nxs"
    DATA_PATH = "/entry1/tomo_entry/data/data"
    assert parse_data(INPUT_FILE, DATA_PATH) == DataConfig(
        in_file=Path(INPUT_FILE),
        data_path=DATA_PATH,
    )


@pytest.mark.parametrize(
    "data_config, image_key_path, config, expected_output",
    [
        (
            DataConfig(Path("/some/path/to/data.nxs"), "/entry1/tomo_entry/data/data"),
            "/entry1/tomo_entry/data/image_key",
            None,
            DarksFlatsFileConfig(
                file=Path("/some/path/to/data.nxs"),
                data_path="/entry1/tomo_entry/data/data",
                image_key_path="/entry1/tomo_entry/data/image_key",
            ),
        ),
        (
            DataConfig(Path("/some/path/to/data.nxs"), "/entry1/tomo_entry/data/data"),
            None,
            {
                "file": "/some/other/path/to/data.h5",
                "data_path": "/data",
                "image_key_path": None,
            },
            DarksFlatsFileConfig(
                file=Path("/some/other/path/to/data.h5"),
                data_path="/data",
                image_key_path=None,
            ),
        ),
        (
            DataConfig(Path("/some/path/to/data.nxs"), "/entry1/tomo_entry/data/data"),
            "/path/to/keys/data_one",
            {
                "file": "/some/path/to/data2.nxs",
                "data_path": "/data",
                "image_key_path": "/path/to/keys/data_two",
            },
            DarksFlatsFileConfig(
                file=Path("/some/path/to/data2.nxs"),
                data_path="/data",
                image_key_path="/path/to/keys/data_two",
            ),
        ),
    ],
    ids=[
        "darks/flats-in-input-file",
        "darks/flats-in-separate-file",
        "darks/flats-in-separate-file-with-image-key",
    ],
)
def test_parse_darks_flats_(
    data_config: DataConfig,
    image_key_path: Optional[str],
    config: Optional[DarksFlatsParam],
    expected_output: DarksFlatsFileConfig,
):
    assert parse_darks_flats(data_config, image_key_path, config) == expected_output


def test_find_tomo_entry_raises_error_if_group_doesnt_exist():
    data_path = (
        Path(__file__).parent
        / "test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs"
    )
    with pytest.raises(ValueError) as e:
        find_tomo_entry(data_path)

    assert f"No NXtomo entry detected in {data_path}" in str(e)


@pytest.mark.parametrize(
    "input_file, config, expected_data_config, expected_image_key_path,  expected_angles_config, expected_darks_config, expected_flats_config",
    [
        (
            Path(__file__).parent / "test_data/tomo_standard.nxs",
            {
                "data_path": "/entry1/tomo_entry/data/data",
                "image_key_path": "/entry1/tomo_entry/instrument/detector/image_key",
                "rotation_angles": {
                    "data_path": "/entry1/tomo_entry/data/rotation_angle"
                },
            },
            DataConfig(
                in_file=Path(__file__).parent / "test_data/tomo_standard.nxs",
                data_path="/entry1/tomo_entry/data/data",
            ),
            "/entry1/tomo_entry/instrument/detector/image_key",
            RawAngles(data_path="/entry1/tomo_entry/data/rotation_angle"),
            DarksFlatsFileConfig(
                file=Path(__file__).parent / "test_data/tomo_standard.nxs",
                data_path="/entry1/tomo_entry/data/data",
                image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
            ),
            DarksFlatsFileConfig(
                file=Path(__file__).parent / "test_data/tomo_standard.nxs",
                data_path="/entry1/tomo_entry/data/data",
                image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
            ),
        ),
        (
            Path(__file__).parent / "test_data/tomo_standard.nxs",
            {"data_path": "auto", "image_key_path": "auto", "rotation_angles": "auto"},
            DataConfig(
                in_file=Path(__file__).parent / "test_data/tomo_standard.nxs",
                data_path="/entry1/tomo_entry/data/data",
            ),
            "/entry1/tomo_entry/instrument/detector/image_key",
            RawAngles(data_path="/entry1/tomo_entry/data/rotation_angle"),
            DarksFlatsFileConfig(
                file=Path(__file__).parent / "test_data/tomo_standard.nxs",
                data_path="/entry1/tomo_entry/data/data",
                image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
            ),
            DarksFlatsFileConfig(
                file=Path(__file__).parent / "test_data/tomo_standard.nxs",
                data_path="/entry1/tomo_entry/data/data",
                image_key_path="/entry1/tomo_entry/instrument/detector/image_key",
            ),
        ),
        (
            Path(__file__).parent / "test_data/k11_diad/k11-18014.nxs",
            {
                "data_path": "/entry/imaging/data",
                "image_key_path": "/entry/instrument/imaging/image_key",
                "rotation_angles": {"data_path": "/entry/imaging_sum/gts_theta_value"},
            },
            DataConfig(
                in_file=Path(__file__).parent / "test_data/k11_diad/k11-18014.nxs",
                data_path="/entry/imaging/data",
            ),
            "/entry/instrument/imaging/image_key",
            RawAngles(data_path="/entry/imaging_sum/gts_theta_value"),
            DarksFlatsFileConfig(
                file=Path(__file__).parent / "test_data/k11_diad/k11-18014.nxs",
                data_path="/entry/imaging/data",
                image_key_path="/entry/instrument/imaging/image_key",
            ),
            DarksFlatsFileConfig(
                file=Path(__file__).parent / "test_data/k11_diad/k11-18014.nxs",
                data_path="/entry/imaging/data",
                image_key_path="/entry/instrument/imaging/image_key",
            ),
        ),
        (
            Path(__file__).parent
            / "test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs",
            {
                "data_path": "/1-TempPlugin-tomo/data",
                "rotation_angles": {
                    "user_defined": {
                        "start_angle": 0,
                        "stop_angle": 180,
                        "angles_total": 724,
                    }
                },
                "darks": {
                    "file": str(
                        Path(__file__).parent
                        / "test_data/i12/separate_flats_darks/dark_field.h5"
                    ),
                    "data_path": "/1-NoProcessPlugin-tomo/data",
                },
                "flats": {
                    "file": str(
                        Path(__file__).parent
                        / "test_data/i12/separate_flats_darks/flat_field.h5"
                    ),
                    "data_path": "/1-NoProcessPlugin-tomo/data",
                },
            },
            DataConfig(
                in_file=Path(__file__).parent
                / "test_data/i12/separate_flats_darks/i12_dynamic_start_stop180.nxs",
                data_path="/1-TempPlugin-tomo/data",
            ),
            None,
            UserDefinedAngles(start_angle=0, stop_angle=180, angles_total=724),
            DarksFlatsFileConfig(
                file=Path(__file__).parent
                / "test_data/i12/separate_flats_darks/dark_field.h5",
                data_path="/1-NoProcessPlugin-tomo/data",
                image_key_path=None,
            ),
            DarksFlatsFileConfig(
                file=Path(__file__).parent
                / "test_data/i12/separate_flats_darks/flat_field.h5",
                data_path="/1-NoProcessPlugin-tomo/data",
                image_key_path=None,
            ),
        ),
    ],
    ids=[
        "test-data-manual",
        "test-data-auto",
        "diad-manual",
        "i12-manual-separate-darks-flats",
    ],
)
def test_parse_loader_config(
    input_file: Path,
    config: Dict[str, Any],
    expected_data_config: DataConfig,
    expected_image_key_path: Optional[str],
    expected_angles_config: AnglesConfig,
    expected_darks_config: DarksFlatsFileConfig,
    expected_flats_config: DarksFlatsFileConfig,
):
    (data_config, image_key_path, angles_config, darks_config, flats_config) = (
        parse_config(input_file, config)
    )
    assert data_config == expected_data_config
    assert image_key_path == expected_image_key_path
    assert angles_config == expected_angles_config
    assert darks_config == expected_darks_config
    assert flats_config == expected_flats_config
