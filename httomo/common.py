import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, unique
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from httomo.utils import Pattern


@dataclass
class MethodFunc:
    """
    Class holding information about each tomography pipeline method

    Parameters
    ==========

    module_name : str
        Fully qualified name of the module where the method is. E.g. httomolib.prep.normalize
    method_func : Callable
        The actual method callable
    wrapper_func: Optional[Callable]
        The wrapper function to handle the execution. It may be None,
        for example for loaders.
    calc_max_slices: Optional[Callable]
        A callable with the signature
        (slice_dim: int, other_dims: int, dtype: np.dtype, available_memory: int) -> int
        which determines the maximum number of slices it can fit in the given memory.
        If it is not present (None), the method is assumed to fit any amount of slices.
    parameters : Dict[str, Any]
        The method parameters that are specified in the pipeline yaml file.
        They are used as kwargs when the method is called.
    cpu : bool
        Whether CPU execution is supported.
    gpu : bool
        Whether GPU execution is supported.
    return_numpy : bool
        Returns numpy array if set to True.
    """

    module_name: str
    method_func: Callable
    wrapper_func: Optional[Callable] = None
    calc_max_slices: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    pattern: Pattern = Pattern.projection
    cpu: bool = True
    gpu: bool = False
    is_loader: bool = False
    return_numpy: bool = False


@dataclass
class ResliceInfo:
    """
    Class holding information regarding reslicing

    Parameters
    ==========

    count: int
        Counter for how many reslices were done so far
    has_warn_printed : bool
        Whether the reslicing warning has been printed
    reslice_dir : Optional[Path]
        The directory to use with file-based reslicing. If None,
        reslicing will be done in-memory.
    """

    count: int
    has_warn_printed: bool
    reslice_dir: Optional[Path] = None


@dataclass
class PlatformSection:
    """
    Data class to represent a section of the pipeline that runs on the same platform.
    That is, all methods contained in this section of the pipeline run either all on CPU
    or all on GPU.

    This is used to iterate through GPU memory in chunks.

    Attributes
    ----------
    gpu : bool
        Whether this section is a GPU section (True) or CPU section (False)
    pattern : Pattern
        To denote the slicing pattern - sinogram, projection
    max_slices : int
        Holds information about how many slices can be fit in one chunk without
        exhausting memory (relevant on GPU only)
    methods : List[MethodFunc]
        List of methods in this section
    output_stats : Tuple[int, int, float, float]
        A tuple containing the min, max, mean, and standard deviation of the
        output of the final method in the section
    """

    gpu: bool
    pattern: Pattern
    max_slices: int
    methods: List[MethodFunc]
    output_stats: Tuple[int, int, float, float]


@dataclass
class RunMethodInfo:
    """
    Class holding information about each method before/while it runs.

    Parameters
    ==========

    dict_params_method : Dict
        The dict of param names and their values for a given method function.
    data_in : str
        The name of the input dataset
    data_out : Union[str, List[str]]
        The name(s) of the output dataset(s)
    dict_httomo_params : Dict
        Dict containing extra params unrelated to wrapped packages but related to httomo
    save_result : bool
        save the result into intermediate dataset
    task_idx: int
        Index of the task in the pipeline being run
    package_name: str
        The name of the package the method is imported from
    method_name: str
        The name of the method being executed
    """

    dict_params_method: Dict[str, Any] = field(default_factory=dict)
    data_in: str = field(default_factory=str)
    data_out: Union[str, List[str]] = field(default_factory=str)
    dict_httomo_params: Dict[str, Any] = field(default_factory=dict)
    save_result: bool = False
    task_idx: int = -1
    package_name: str = None
    method_name: str = None
