from typing import Literal, Optional, TypeAlias, TypedDict, Union
from httomo.loaders.standard_tomo_loader import (
    AnglesConfig,
    RawAngles,
    UserDefinedAngles,
)

from httomo.preview import PreviewConfig, PreviewDimConfig


class StartStopEntry(TypedDict):
    start: Optional[int]
    stop: Optional[int]


PreviewParamEntry: TypeAlias = Union[Literal["mid"], StartStopEntry]


class PreviewParam(TypedDict):
    angles: Optional[StartStopEntry]
    detector_y: Optional[PreviewParamEntry]
    detector_x: Optional[PreviewParamEntry]


def parse_preview(
    param_value: Optional[PreviewParam],
    data_shape: tuple[int, int, int],
) -> PreviewConfig:
    DIMENSION_MAPPINGS: dict[str, int] = {"angles": 0, "detector_y": 1, "detector_x": 2}

    if param_value is None:
        param_value = {"angles": None, "detector_y": None, "detector_x": None}

    dims = param_value.keys()
    for dim in DIMENSION_MAPPINGS.keys():
        if dim not in dims:
            param_value[dim] = None

    if param_value["angles"] == "mid":
        raise ValueError("'mid' keyword not supported for angles dimension")

    dim_configs = {}
    for dim, slice_info in param_value.items():
        if slice_info is None:
            start = 0
            stop = data_shape[DIMENSION_MAPPINGS[dim]]
        elif slice_info == "mid":
            start, stop = _get_middle_slice_indices(data_shape[DIMENSION_MAPPINGS[dim]])
        else:
            start = slice_info.get("start", 0)
            stop = slice_info.get("stop", data_shape[DIMENSION_MAPPINGS[dim]])

        dim_configs[dim] = PreviewDimConfig(start=start, stop=stop)

    return PreviewConfig(
        angles=dim_configs["angles"],
        detector_y=dim_configs["detector_y"],
        detector_x=dim_configs["detector_x"],
    )


def _get_middle_slice_indices(dim_len: int) -> tuple[int, int]:
    """
    Get indices for middle 3 slices of either the detector_y or detetcor_x dimension.

    For even length dimensions, will return the 4 indices corresponding to:
    - dim_len // 2 - 2
    - dim_len // 2 - 1
    - dim_len // 2
    - dim_len // 2 + 1

    For odd length dimensions, will return the 3 indices corresponding to:
    - dim_len // 2 - 1
    - dim_len // 2
    - dim_len // 2 + 1
    """
    mid_slice_idx = dim_len // 2

    if mid_slice_idx == 1:
        return 0, dim_len

    if dim_len % 2 == 0:
        return mid_slice_idx - 2, mid_slice_idx + 1
    else:
        return mid_slice_idx - 1, mid_slice_idx + 1


class RawAnglesParam(TypedDict):
    data_path: str


class UserDefinedAnglesParamInner(TypedDict):
    start_angle: int
    stop_angle: int
    angles_total: int


class UserDefinedAnglesParam(TypedDict):
    user_defined: UserDefinedAnglesParamInner


AnglesParam: TypeAlias = Union[RawAnglesParam, UserDefinedAnglesParam]


def parse_angles(angles_data: AnglesParam) -> AnglesConfig:
    keys = angles_data.keys()

    if "data_path" in keys:
        return RawAngles(data_path=angles_data["data_path"])

    if "user_defined" in keys:
        return UserDefinedAngles(
            start_angle=angles_data["user_defined"]["start_angle"],
            stop_angle=angles_data["user_defined"]["stop_angle"],
            angles_total=angles_data["user_defined"]["angles_total"],
        )

    raise ValueError(f"Unknown rotation_angles param value for loader: {angles_data}")
