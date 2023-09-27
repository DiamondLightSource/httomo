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
        Fully qualified name of the module where the method is. E.g. httomolibgpu.prep.normalize
    method_func : Callable
        The actual method callable
    wrapper_func: Optional[Callable]
        The wrapper function to handle the execution. It may be None,
        for example for loaders.
    calc_max_slices: Optional[Dict[str, Any]]
        None for the CPU method or Dictionary with parameters for GPU memory estimation.
        This determines the maximum number of slices it can fit in the given memory.
        If it is not present (None), the method is assumed to fit any amount of slices.
    output_dims_change : Dict[str, Any]
        False - the output data dimensions of the method are the same as input
        True - the data dimensions are different with respect to input data dimensions.
    parameters : Dict[str, Any]
        The method parameters that are specified in the pipeline yaml file.
        They are used as kwargs when the method is called.
    cpu : bool
        Whether CPU execution is supported.
    gpu : bool
        Whether GPU execution is supported.
    cupyrun : bool
        Whether CuPy API is used.
    return_numpy : bool
        Returns numpy array if set to True.
    idx_global: int
        A global index of the method in the pipeline.
    global_statistics: bool
        Whether global statistics needs to be calculated on the output of the method.
    """

    module_name: str
    method_func: Callable
    wrapper_func: Optional[Callable] = None
    calc_max_slices: Optional[Dict[str, Any]] = None
    output_dims_change: Dict[str, Any] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    pattern: Pattern = Pattern.projection
    cpu: bool = True
    gpu: bool = False
    cupyrun: bool = False
    return_numpy: bool = False
    idx_global: int = 0
    global_statistics: bool = False


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
    Data class to represent a section of the pipeline. Section can combine methods
    if they run on the same platform (cpu or gpu) and have the same pattern. 
    The sections can be further divided if necessary if the results of the method
    needed to be saved. 
    NOTE: More fine division of sections into subsections will slow down 
    the pipeline.

    Mainly used to iterate through GPU memory in chunks.

    Attributes
    ----------
    gpu : bool
        Whether this section is a GPU section (True) or CPU section (False)
    pattern : Pattern
        To denote the slicing pattern - sinogram, projection
    reslice : bool
        This tells the runner if we need to reslice the data before next section
    max_slices : int
        Holds information about how many slices can be fit in one chunk without
        exhausting memory (relevant on GPU only)
    methods : List[MethodFunc]
        List of methods in this section
    """

    gpu: bool
    pattern: Pattern
    reslice: bool
    max_slices: int
    methods: List[MethodFunc]


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
        Index of the local task in the section being run
    task_idx_global: int 
        Index of the global task (method) in the pipeline
    package_name: str
        The name of the package the method is imported from
    method_name: str
        The name of the method being executed
    global_statistics: bool
        Whether global statistics needs to be calculated on the output of the method.        
    """

    dict_params_method: Dict[str, Any] = field(default_factory=dict)
    data_in: str = field(default_factory=str)
    data_out: Union[str, List[str]] = field(default_factory=str)
    dict_httomo_params: Dict[str, Any] = field(default_factory=dict)
    save_result: bool = False
    task_idx: int = -1
    task_idx_global: int = -1
    package_name: str = None
    method_name: str = None
    global_statistics: bool = False
