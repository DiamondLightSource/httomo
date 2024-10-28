"""
Types that represent python dicts for angles and preview configuration which are generated from
parsing a pipeline file into python, and functions to transform these python dicts to internal
types that loaders can use.
"""

from typing import Literal, Optional, TypeAlias, TypedDict, Union

from httomo.loaders.types import (
    AnglesConfig,
    RawAngles,
    UserDefinedAngles,
)
from httomo.preview import PreviewConfig, PreviewDimConfig


class StartStopEntry(TypedDict):
    """
    Configuration for a single dimension's previewing in terms of start/stop values.
    """

    start: Optional[int]
    stop: Optional[int]


PreviewParamEntry: TypeAlias = Union[Literal["mid"], StartStopEntry]


class PreviewParam(TypedDict):
    """
    Preview configuration dict.
    """

    angles: Optional[StartStopEntry]
    detector_y: Optional[PreviewParamEntry]
    detector_x: Optional[PreviewParamEntry]


PreviewKeys = Literal["angles", "detector_y", "detector_x"]


def parse_preview(
    param_value: Optional[PreviewParam],
    data_shape: tuple[int, int, int],
) -> PreviewConfig:
    """
    Convert python dict representing preview information generated from parsing the
    pipeline file, into an internal preview configuration type that loaders can use.

    Parameters
    ----------
    param_value : Optional[PreviewParam]
        The python dict parsed from the pipeline file that represents the preview configuration
        in the loader.

    data_shape : tuple[int, int, int]
        The shape of the 3D input data.

    Returns
    -------
    PreviewConfig
        Preview configuration that loaders can use.
    """
    DIMENSION_MAPPINGS: dict[PreviewKeys, int] = {
        "angles": 0,
        "detector_y": 1,
        "detector_x": 2,
    }

    if param_value is None:
        param_value = {"angles": None, "detector_y": None, "detector_x": None}

    dims = param_value.keys()
    for dim in DIMENSION_MAPPINGS.keys():
        if dim not in dims:
            param_value[dim] = None

    if param_value["angles"] == "mid":
        raise ValueError("'mid' keyword not supported for angles dimension")

    def conv_param(par: Optional[PreviewParamEntry], length: int):
        if par is None:
            start = 0
            stop = length
        elif par == "mid":
            start, stop = _get_middle_slice_indices(length)
        else:
            val = par.get("start", None)
            start = 0 if val is None else val
            val = par.get("stop", None)
            stop = length if val is None else val

        return PreviewDimConfig(start=start, stop=stop)

    return PreviewConfig(
        angles=conv_param(param_value["angles"], data_shape[0]),
        detector_y=conv_param(param_value["detector_y"], data_shape[1]),
        detector_x=conv_param(param_value["detector_x"], data_shape[2]),
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
    """
    Angles configuration dict for when the rotation angle values are in a dataset within the
    input NeXuS/hdf5 file.
    """

    data_path: str


class UserDefinedAnglesParamInner(TypedDict):
    """
    Start, stop, and total angles configuration to generate the rotation angle values.
    """

    start_angle: int
    stop_angle: int
    angles_total: int


class UserDefinedAnglesParam(TypedDict):
    """
    Angles configuration dict for when the rotation angle values are manually defined (rather
    than taken from the input NeXuS/hdf5 file).
    """

    user_defined: UserDefinedAnglesParamInner


AnglesParam: TypeAlias = Union[RawAnglesParam, UserDefinedAnglesParam]


def parse_angles(angles_data: AnglesParam) -> AnglesConfig:
    """
    Convert python dict representing angles information generated from parsing the
    pipeline file, into an internal angles configuration type that loaders can use.

    Parameters
    ----------
    angles_data : AnglesParam
        The python dict parsed from the pipeline file that represents the angles configuration
        in the loader.

    Returns
    -------
    AnglesConfig
        Angles configuration that loaders can use.
    """
    if "data_path" in angles_data and isinstance(angles_data["data_path"], str):
        return RawAngles(data_path=angles_data["data_path"])

    if (
        "user_defined" in angles_data
        and "start_angle" in angles_data["user_defined"]
        and "stop_angle" in angles_data["user_defined"]
        and "angles_total" in angles_data["user_defined"]
    ):
        return UserDefinedAngles(
            start_angle=angles_data["user_defined"]["start_angle"],
            stop_angle=angles_data["user_defined"]["stop_angle"],
            angles_total=angles_data["user_defined"]["angles_total"],
        )

    raise ValueError(f"Unknown rotation_angles param value for loader: {angles_data}")
