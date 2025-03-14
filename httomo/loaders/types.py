"""
Loader configuration types.
"""

from pathlib import Path
from typing import NamedTuple, TypeAlias, Union


class RawAngles(NamedTuple):
    """
    Configure rotation angle values to come from a dataset within the input NeXuS/hdf5 file.
    """

    data_path: str


class UserDefinedAngles(NamedTuple):
    """
    Configure rotation angle values to be generated by specifying a start angle, stop angle,
    and the total number of angles.
    """

    start_angle: int
    stop_angle: int
    angles_total: int


AnglesConfig: TypeAlias = Union[RawAngles, UserDefinedAngles]


class DataConfig(NamedTuple):
    """
    Input data configuration.
    """

    in_file: Path
    data_path: str
