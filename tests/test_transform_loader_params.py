import pytest

from httomo.preview import PreviewConfig, PreviewDimConfig
from httomo.transform_loader_params import PreviewParam, parse_preview


@pytest.mark.parametrize(
    "data_shape, preview_param_value, expected_preview_config",
    [
        (
            (220, 128, 160),
            [None, None, None],
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            [None],
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            [
                {"start": 0, "stop": 220},
                {"start": 0, "stop": 128},
                None,
            ],
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=128),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 128, 160),
            [
                {"start": 0, "stop": 220},
                "mid",
                None,
            ],
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=62, stop=65),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 127, 160),
            [
                {"start": 0, "stop": 220},
                "mid",
                None,
            ],
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=62, stop=64),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
        (
            (220, 3, 160),
            [
                {"start": 0, "stop": 220},
                "mid",
                None,
            ],
            PreviewConfig(
                angles=PreviewDimConfig(start=0, stop=220),
                detector_y=PreviewDimConfig(start=0, stop=3),
                detector_x=PreviewDimConfig(start=0, stop=160),
            ),
        ),
    ],
    ids=[
        "all_dims_none",
        "single_none",
        "angles_det_y_not_none",
        "det_y_even_len_get_mid",
        "det_y_odd_len_get_mid",
        "det_y_small_len_get_mid",
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
    PREVIEW_PARAM_VALUE: PreviewParam = ["mid"]
    with pytest.raises(ValueError):
        _ = parse_preview(PREVIEW_PARAM_VALUE, DATA_SHAPE)
