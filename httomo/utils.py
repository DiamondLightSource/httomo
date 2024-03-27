from contextlib import contextmanager
import re
from enum import Enum
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Dict, List, Literal, Tuple

from mpi4py.MPI import Comm
import numpy as np

import httomo.globals
from httomo.data import mpiutil

gpu_enabled = False
try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
        gpu_enabled = True  # CuPy is installed and GPU is available
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp

        print("CuPy is installed but GPU device inaccessible")

except ImportError:
    import numpy as xp

    print("CuPy is not installed")


class Colour:
    """
    Class for storing the ANSI escape codes for different colours.
    """

    LIGHT_BLUE = "\033[1;34m"
    LIGHT_BLUE_BCKGR = "\033[1;44m"
    BLUE = "\33[94m"
    CYAN = "\33[96m"
    GREEN = "\33[92m"
    YELLOW = "\33[93m"
    MAGENTA = "\33[95m"
    RED = "\33[91m"
    END = "\033[0m"
    BVIOLET = "\033[1;35m"
    LYELLOW = "\033[33m"
    BACKG_RED = "\x1b[6;37;41m"


def log_once(output: Any, comm: Comm, colour: Any = Colour.GREEN, level=0) -> None:
    """
    Log output to console and log file if the process is rank zero.

    Parameters
    ----------
    output : Any
        The item to be printed.
    comm : Comm
        The comm used to determine the rank zero process.
    colour : str, optional
        The colour of the output.
    level : int, optional
        The level of the log message. 0 is info, 1 is debug, 2 is warning.
    """
    if mpiutil.rank == 0:
        if isinstance(output, list):
            output = "".join(
                [f"{colour}{out}{Colour.END}" for out, colour in zip(output, colour)]
            )
        else:
            output = f"{colour}{output}{Colour.END}"

        if httomo.globals.logger is not None:
            if level == 1:
                httomo.globals.logger.debug(output)
            elif level == 2:
                httomo.globals.logger.warn(output)
            else:
                httomo.globals.logger.info(output)


def log_rank(output: Any, comm: Comm) -> None:
    """
    Log output to log file with the process rank.

    Parameters
    ----------
    output : Any
        The item to be printed.
    comm : Comm
        The comm used to determine the process rank.
    """
    if httomo.globals.logger is not None:
        httomo.globals.logger.debug(f"RANK: [{comm.rank}], {output}")


def log_exception(output: str, colour: str = Colour.RED) -> None:
    """
    Log an exception to the log file.

    Parameters
    ----------
    output : str
        The exception to be logged.
    """
    if httomo.globals.logger is not None:
        httomo.globals.logger.error(f"{colour}{output}{Colour.END}")

    #: now this will cause the pipeline to crash
    #: remove ansi escape sequences from the log file
    remove_ansi_escape_sequences(f"{httomo.globals.run_out_dir}/user.log")


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
        elif slice_info == "mid":
            #  user can simply write 'mid' to get 3 slices around the middle section of a chosen dimension
            mid_slice = data_shape[idx] // 2
            if mid_slice > 1:
                preview_str += f"{mid_slice-2}:{mid_slice+1}:{1}"
            else:
                # the size of the dimension is less than 4 so all data can be taken
                preview_str += ":"
        else:
            start = slice_info.get("start", None)
            stop = slice_info.get("stop", None)
            step = slice_info.get("step", None)

            warn_on = False
            if start is not None and (start < 0 or start >= data_shape[idx]):
                warn_on = True
                str_warn = (
                    f"The 'start' preview {start} is outside the data dimension range"
                    + f" from 0 to {data_shape[idx]}"
                )
            if stop is not None and (stop < 0 or stop >= data_shape[idx]):
                warn_on = True
                str_warn = (
                    f"The 'stop' preview {start} is outside the data dimension range"
                    + f" from 0 to {data_shape[idx]}"
                )
            if step is not None and step < 0:
                warn_on = True
                str_warn = "The 'step' in preview cannot be negative"

            if warn_on:
                log_exception(str_warn, colour=Colour.BACKG_RED)
                raise ValueError("Preview error: " + str_warn)

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


def _get_slicing_dim(pattern: Pattern) -> Literal[1, 2]:
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
        # this pattern should inherit the pattern from the previous section if it is available.
        # It needs taken care of in the runner.
        return 1
    else:
        err_str = f"An unknown pattern has been encountered {pattern}"
        log_exception(err_str)
        raise ValueError(err_str)


def get_data_in_data_out(method_name: str, dict_params_method: Dict[str, Any]) -> tuple:
    """
    Get the input and output datasets in a list
    """
    if (
        "data_in" in dict_params_method.keys()
        and "data_out" in dict_params_method.keys()
    ):
        data_in = dict_params_method.pop("data_in")
        data_out = dict_params_method.pop("data_out")
    else:
        # TODO: This error reporting is possibly better handled by
        # schema validation of the user config YAML
        if method_name != "save_to_images":
            if (
                "data_in" in dict_params_method.keys()
                and "data_out" not in dict_params_method.keys()
            ):
                # Assume "data_out" to be the same as "data_in"
                data_in = [dict_params_method.pop("data_in")]
                data_out = data_in
            else:
                err_str = "Invalid in/out dataset parameters"
                log_exception(err_str)
                raise ValueError(err_str)
        else:
            data_in = dict_params_method.pop("data_in")
            data_out = None

    return data_in, data_out


def remove_ansi_escape_sequences(filename):
    """
    Remove ANSI escape sequences from a file.
    """
    ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")    
    if Path(filename).exists():
        with open(filename, "r") as f:
            lines = f.readlines()
        with open(filename, "w") as f:
            for line in lines:
                f.write(ansi_escape.sub("", line))


class catchtime:
    """A context manager to measure ns-accurate time for the context block.
    
    Usage:
        with catchtime() as t:
            ...
            
        print(t.elapsed)
    """
    def __enter__(self):
        self.start = perf_counter_ns()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (perf_counter_ns() - self.start) * 1e-9
    

class catch_gputime:
    def __enter__(self):
        if gpu_enabled:
            self.start = xp.cuda.Event()
            self.start.record()
        return self
            
    def __exit__(self, type, value, traceback):
        if gpu_enabled:
            self.end = xp.cuda.Event()
            self.end.record()
    
    @property
    def elapsed(self) -> float:
        if gpu_enabled:
            self.end.synchronize()
            return xp.cuda.get_elapsed_time(self.start, self.end) * 1e-3
        else:
            return 0.0


def make_3d_shape_from_shape(shape: List[int]) -> Tuple[int, int, int]:
    """Given a shape as a list of length 3, return a corresponding tuple 
       with the right typing type (required to make mypy type checks work)
    """
    assert len(shape) == 3, "3D shape expected"
    return (shape[0], shape[1], shape[2])


def make_3d_shape_from_array(array: np.ndarray) -> Tuple[int, int, int]:
    """Given a 3D array, return a corresponding shape tuple
       with the right typing type (required to make mypy type checks work)
    """
    return make_3d_shape_from_shape(list(array.shape))
