from enum import IntEnum, unique


@unique
class PipelineTasks(IntEnum):
    """An enumeration of available pipeline stages."""

    LOAD = 0
    FILTER = 1
    NORMALIZE = 2
    STRIPES = 3
    CENTER = 4
    RESLICE = 5
    RECONSTRUCT = 6
    SAVE = 7
