import re
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


def remove_ansi_escape_sequences(filename):
    """
    Remove ANSI escape sequences from a file.
    """
    ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        for line in lines:
            f.write(ansi_escape.sub("", line))
