from typing import Any
from mpi4py.MPI import Comm
from typing import Tuple, List, Dict, Callable
from enum import Enum


class Colour:
    """
    Class for storing the ANSI escape codes for different colours.
    """

    LIGHT_BLUE = "\033[1;34m"
    BLUE = "\33[94m"
    CYAN = "\33[96m"
    GREEN = "\33[92m"
    YELLOW = "\33[93m"
    MAGENTA = "\33[95m"
    RED = "\33[91m"
    END = "\033[0m"
    BVIOLET = "\033[1;35m"


def print_once(output: Any, comm: Comm, colour: Any = Colour.GREEN) -> None:
    """
    Print an output from rank zero only.

    Parameters
    ----------
    output : Any
        The item to be printed.
    comm : Comm
        The comm used to determine the rank zero process.
    colour : str, optional
        The colour of the output.
    """
    if comm.rank == 0:
        if isinstance(output, list):
            output = "".join(
                [f"{colour}{out}{Colour.END}" for out, colour in zip(output, colour)]
            )
            print(output)
        else:
            print(colour + output + Colour.END)


def print_rank(output: Any, comm: Comm) -> None:
    """Print an output with rank prefix.

    Parameters
    ----------
    output : Any
        The item to be printed.
    comm : Comm
        The comm used to determine the process rank.
    """
    print(f"RANK: [{comm.rank}], {output}")


def _parse_preview(
    preview: List[Dict[str, int]], data_shape: Tuple[int], data_indices: List[int]
) -> str:
    """Parse the python list that represents the preview parameter in the loader
    into a string that the helper loader functions in
    `httomo.data.hdf._utils.load` can understand.

    Parameters
    ----------
    preview : List[Dict[str, int]]
        A list of dicts, where each dict represents the start, stop, step values
        that a dimension of data can be previewed/sliced.

    data_shape : Tuple[int]
        The shape of the original data to apply previewing to.

    data_indices : List[int]
        The indices where projection data is in the dataset.

    Returns
    -------
    str
        A string that represents the preview parameter from the YAML config.
    """
    preview_str = ""

    # Pad the `preview` list with None until it is the same length as the number
    # of dimensions in the data, since the user may not have specified a preview
    # value for every dimension after the last dimension with previewing in the
    # YAML config
    if len(preview) < len(data_shape):
        while len(preview) < len(data_shape):
            preview.append(None)

    for idx, slice_info in enumerate(preview):
        if slice_info is None:
            if idx == 0:
                # TODO: Assuming data to be stack of projections, so dim 0 would
                # be the rotation angle dim, and as such would contain flats and
                # darks. These should be cropped out by default if no explicit
                # previewing is specified.
                #
                # However, assumption perhaps should be removed to make this
                # function more generic.
                preview_str += f"{data_indices[0]}:{data_indices[-1]+1}"
            else:
                preview_str += ":"
        elif slice_info == 'mid':
            #  user can simply write 'mid' to get 3 slices around the middle section of a choosen dimension
            mid_slice = data_shape[idx] // 2
            if mid_slice > 1:
                preview_str += f"{mid_slice-2}:{mid_slice+1}:{1}"
            else:
                # the size of the dimension is less than 4 so all data can be taken
                preview_str += ":"
        else:
            start = slice_info["start"] if "start" in slice_info.keys() else None
            stop = slice_info["stop"] if "stop" in slice_info.keys() else None
            step = slice_info["step"] if "step" in slice_info.keys() else None
            start_str = f"{start if start is not None else ''}"
            stop_str = f"{stop if stop is not None else ''}"
            step_str = f"{step if step is not None else ''}"
            preview_str += f"{start_str}:{stop_str}:{step_str}"

        # Depending on if this is the last dimension in the data or not, a comma
        # may or may not be needed
        if idx < len(preview) - 1:
            preview_str += ", "

    return preview_str


class Pattern(Enum):
    """Enum for the different slicing-orientations/"patterns" that tomographic
    data can have.
    """

    projection = 0
    sinogram = 1
    all = 2


def _get_slicing_dim(pattern: Pattern) -> int:
    """Assuming 3D projection data with the axes ordered as
    - axis 0 = rotation angle
    - axis 1 = detector y
    - axis 2 = detector x

    when given the pattern of a method, return the dimension of the data that
    the method requires the data to be sliced in.
    """
    if pattern == Pattern.projection:
        return 1
    elif pattern == Pattern.sinogram:
        return 2
    elif pattern == Pattern.all:
        # Any slicing dimension is fine, so arbitrarily just return 1
        return 1
    else:
        err_str = f"An unknown pattern has been encountered {pattern}"
        raise ValueError(err_str)
