from typing import NamedTuple, TypeAlias, Union


class RawAngles(NamedTuple):
    data_path: str


class UserDefinedAngles(NamedTuple):
    start_angle: int
    stop_angle: int
    angles_total: int


AnglesConfig: TypeAlias = Union[RawAngles, UserDefinedAngles]
