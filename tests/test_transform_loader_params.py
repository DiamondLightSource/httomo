import pytest

from httomo.loaders.types import (
    AnglesConfig,
    RawAngles,
    UserDefinedAngles,
)
from httomo.preview import PreviewConfig, PreviewDimConfig
from httomo.transform_loader_params import (
    PreviewParam,
    parse_angles,
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
