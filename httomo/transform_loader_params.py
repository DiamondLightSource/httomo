from typing import Literal, TypeAlias, TypedDict, Union

from httomo.preview import PreviewConfig, PreviewDimConfig


class StartStopEntry(TypedDict):
    start: int
    stop: int


PreviewParamEntry: TypeAlias = Union[None, Literal["mid"], StartStopEntry]


# TODO: Need to disallow "mid" from being the 0th entry in the list
PreviewParam: TypeAlias = list[PreviewParamEntry]


def parse_preview(
    param_value: PreviewParam,
    data_shape: tuple[int, int, int],
) -> PreviewConfig:
    preview_data = param_value[:]
    if len(param_value) < len(data_shape):
        preview_data += [None] * (len(data_shape) - len(preview_data))

    assert len(preview_data) == 3
    dim_configs: list[PreviewDimConfig] = []

    if preview_data[0] == "mid":
        raise ValueError("'mid' keyword not supported for angles dimension")

    for idx, slice_info in enumerate(preview_data):
        if slice_info is None:
            start = 0
            stop = data_shape[idx]
        elif slice_info == "mid":
            start, stop = _get_middle_slice_indices(data_shape[idx])
        else:
            start = slice_info["start"]
            stop = slice_info["stop"]

        dim_configs.append(PreviewDimConfig(start=start, stop=stop))

    return PreviewConfig(
        angles=dim_configs[0],
        detector_y=dim_configs[1],
        detector_x=dim_configs[2],
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
