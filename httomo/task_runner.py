import dataclasses
import multiprocessing
import time
import math
import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from httomolib.misc.images import save_to_images
from mpi4py import MPI
from numpy import ndarray

from httomo.utils import gpu_enabled, xp

import httomo.globals
from httomo._stats.globals import min_max_mean_std
from httomo.common import MethodFunc, PlatformSection, ResliceInfo, RunMethodInfo
from httomo.data.hdf._utils.chunk import get_data_shape, save_dataset
from httomo.data.hdf._utils.reslice import reslice, reslice_filebased
from httomo.data.hdf._utils.save import intermediate_dataset
from httomo.data.hdf.loaders import LoaderData
from httomo.methods_database.query import get_method_info
from httomo.postrun import postrun_method
from httomo.prerun import prerun_method
from httomo.utils import (
    Colour,
    Pattern,
    _get_slicing_dim,
    log_exception,
    log_once,
    remove_ansi_escape_sequences,
)
from httomo.wrappers_class import HttomolibWrapper, HttomolibgpuWrapper, TomoPyWrapper
from httomo.yaml_utils import open_yaml_config


def run_tasks(
    in_file: Path,
    yaml_config: Path,
    dimension: int,
    pad: int = 0,
    ncore: int = 1,
    save_all: bool = False,
    reslice_dir: Optional[Path] = None,
) -> None:
    """Run the pipeline defined in the YAML config file

    Parameters
    ----------
    in_file : Path
        The file to read data from.
    yaml_config : Path
        The file containing the processing pipeline info as YAML.
    dimension : int
        The dimension to slice in.
    pad : int
        The padding size to use. Defaults to 0.
    ncore : int
        The number of the CPU cores per process.
    save_all : bool
        Specifies if intermediate datasets should be saved for all tasks in the
        pipeline.
    reslice_dir : Optional[Path]
        Path where to store the reslice intermediate files, or None if reslicing
        should be done in-memory.
    """
    comm = MPI.COMM_WORLD
    if comm.size == 1:
        # use all available CPU cores if not an MPI run
        ncore = multiprocessing.cpu_count()

    # Define dict to store arrays of the whole pipeline using provided YAML
    # Define list to store dataset stats for each task in the user config YAML
    dict_datasets_pipeline, glob_stats = _initialise_datasets_and_stats(yaml_config)

    # Get a list of the python functions associated to the methods defined in
    # user config YAML
    method_funcs = _get_method_funcs(yaml_config, comm)

    # Define dict of params that are needed by loader functions
    dict_loader_extra_params = {
        "in_file": in_file,
        "dimension": dimension,
        "pad": pad,
        "comm": comm,
    }

    # store info about reslicing with ResliceInfo
    reslice_info = ResliceInfo(
        count=0, has_warn_printed=False, reslice_dir=reslice_dir
    )

    # Associate patterns to method function objects
    for i, method_func in enumerate(method_funcs):
        method_funcs[i] = _assign_pattern_to_method(method_func)

    #: no need to add loader into a platform section
    platform_sections = _determine_platform_sections(method_funcs[1:])

    # Check pipeline for the number of parameter sweeps present. If one is
    # defined, raise an error, due to not supporting parameter sweeps in a
    # "performance" run of httomo
    params = [m.parameters for m in method_funcs]
    no_of_sweeps = sum(map(_check_params_for_sweep, params))

    if no_of_sweeps > 0:
        err_str = (
            f"There exists {no_of_sweeps} parameter sweep(s) in the "
            "pipeline, but parameter sweeps are not supported in "
            "`httomo performance`. Please either:\n  1) Remove the parameter "
            "sweeps.\n  2) Use `httomo preview` to run this pipeline."
        )
        log_exception(err_str)
        raise ValueError(err_str)

    # start MPI timer for rank 0
    if comm.rank == 0:
        start_time = MPI.Wtime()

    #: add to the console and log file, the full path to the user.log file
    log_once(
        f"See the full log file at: {httomo.globals.run_out_dir}/user.log",
        comm,
        colour=Colour.CYAN,
        level=0,
    )
    method_funcs[0].parameters.update(dict_loader_extra_params)

    # Check if a value for the `preview` parameter of the loader has
    # been provided
    if "preview" not in method_funcs[0].parameters.keys():
        method_funcs[0].parameters["preview"] = [None]

    loader_method_name = method_funcs[0].parameters.pop("method_name")
    log_once(
        f"Running task 1 (pattern={method_funcs[0].pattern.name}): {loader_method_name}...",
        comm,
        colour=Colour.LIGHT_BLUE,
        level=0,
    )

    loader_start_time = time.perf_counter_ns()

    # function to be called from httomo.data.hdf.loaders
    loader_func = method_funcs[0].method_func
    # collect meta data from LoaderData.
    loader_info = loader_func(**method_funcs[0].parameters)

    output_str_list = [
        f"    Finished task 1 (pattern={method_funcs[0].pattern.name}): {loader_method_name} (",
        "httomo",
        f") Took {float(time.perf_counter_ns() - loader_start_time)*1e-6:.2f}ms",
    ]
    output_colour_list = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
    log_once(output_str_list, comm=comm, colour=output_colour_list)

    # Update `dict_datasets_pipeline` dict with the data that has been
    # loaded by the loader
    # NOTE: conversion to float32 to avoid creating an additional output array in the GPU loop
    dict_datasets_pipeline[method_funcs[0].parameters["name"]] = np.float32(loader_info.data)
    dict_datasets_pipeline["flats"] = np.float32(loader_info.flats)
    dict_datasets_pipeline["darks"] = np.float32(loader_info.darks)

    # Extra params relevant to httomo that a wrapper function might need
    possible_extra_params = [
        (["darks"], dict_datasets_pipeline["darks"]),
        (["flats"], dict_datasets_pipeline["flats"]),
        (["angles", "angles_radians"], loader_info.angles),
        (["comm"], comm),
        (["out_dir"], httomo.globals.run_out_dir),
        (["return_numpy"], False),
    ]
    # data shape and dtype are useful when calculating max slices
    data_shape = loader_info.data.shape
    data_dtype = loader_info.data.dtype
    
    ##---------- Main GPU loop starts here ------------##
    idx = 0
    # initialise the CPU data array with the loaded data, we override it at the end of every section
    data_full_section = dict_datasets_pipeline[method_funcs[idx].parameters["name"]]
    for i, section in enumerate(platform_sections):
        print(f"Section {i}")
        # determine the max_slices for the whole section
        _update_max_slices(section, data_shape, data_dtype)
        # in order to iterate over max slices we need to know the slicing
        # dimension of the section
        slicing_dim_section = _get_slicing_dim(section.pattern) - 1
        # iterations_for_blocks determines the number of iterations needed
        # to raster through the data in blocks that would fit into the GPU memory
        iterations_for_blocks = math.ceil(data_shape[slicing_dim_section] / section.max_slices)
        
        ##---------- the loop over blocks in a section for a chunk of data ------------##
        indices_start = 0
        indices_end = int(section.max_slices)
        slc_indices = [slice(None)] * len(data_shape)
        for it_blocks in range(iterations_for_blocks):
            # preparing indices for the slicing of the data in blocks
            slc_indices[slicing_dim_section] = slice(indices_start, indices_end, 1)         

            ##---------- the loop over methods in a block ------------##
            for m_ind, methodfunc_sect in enumerate(section.methods):
                # preparing everything for the wrapper execution
                module_path = methodfunc_sect.module_name
                method_name = methodfunc_sect.method_func.__name__
                func_wrapper = methodfunc_sect.wrapper_func
                package_name = methodfunc_sect.module_name.split(".")[0]
                
                print(f"Method {method_name}")
                # Only run the `prerun_method()` once for a method, when the
                # first block is going to be processed. This is because the
                # method's parameters will not have changed from the first time
                # it has run, only its input data, which is taken care of
                # outside the `prerun_method()` function.
                if it_blocks == 0:
                    #: create an object that would be passed along to prerun_method,
                    #: run_method, and postrun_method
                    run_method_info = RunMethodInfo(task_idx=m_ind)

                    #: prerun - before running the method, update the dictionaries
                    prerun_method(
                        run_method_info,
                        section,
                        possible_extra_params,
                        methodfunc_sect,
                        dict_datasets_pipeline,
                    )

                if m_ind == 0:
                    # Assign the block of data on a CPU to `data` parameter of the method
                    # this should happen once in the beginning of the loop over methods
                    run_method_info.dict_httomo_params["data"] = data_full_section[tuple(slc_indices)]
                else:
                    # Initialise with result from previous method
                    run_method_info.dict_httomo_params["data"] = res
                
                # remove methods name from the parameters list of a method
                run_method_info.dict_params_method.pop('method_name')
                
                # run the wrapper
                res = func_wrapper(
                    method_name,
                    run_method_info.dict_params_method,
                    **run_method_info.dict_httomo_params,
                )        
                
                if isinstance(res, (tuple, list)):
                    err_str = (
                        "Methods producing multiple outputs are not yet "
                        "supported in the GPU loop"
                    )
                    raise ValueError(err_str)

                # NOTE: If `save_to_images` needs the stats of the output of the
                # previous section, they can be accessed via
                # platform_sections[i-1].output_stats                
            ##************* Methods loop is complete *************##
            # Saving the processed block. 
            data_full_section[tuple(slc_indices)] = res
            # NOTE: data_block must be on the CPU already, assuming that
            # the last method in the loop will return the numpy array and not cupy

            # re-initialise the slicing indices
            indices_start = indices_end
            # checking if we still within the slicing dimension size and take remaining portion
            res = (indices_start + int(section.max_slices)) - data_shape[slicing_dim_section] 
            if res > 0:
                res = int(section.max_slices) - res
                indices_end += res
            else:
                indices_end += section.max_slices
            # flushing the GPU memory allocated in the methods loop
            if gpu_enabled:
                xp.get_default_memory_pool().free_all_blocks()
                cache = xp.fft.config.get_plan_cache()
                cache.clear()
            
        ##************* Blocks loop is complete *************##
        # TODO: After the blocks loop is complete the full data
        # is in "data_full_section" and the hdf5 file can be saved
        # or resliced if needed

        # Calculate stats on result of the last method in the section
        #
        # NOTE: Currently assuming that the dataset to calculate stats from is
        # the same input dataset from the beginning of the sections loop; should
        # this instead be getting the output dataset from the last method that
        # was in the section which has just completed?
        section.output_stats = min_max_mean_std(data_full_section, comm)

        # Assume that, after every section has been completed (except for the
        # last section), a reslice should occur
        if i < len(platform_sections) - 1:
            next_section_in = platform_sections[i+1].methods[0].parameters["data_in"]
            # TODO: Ensure that the required dataset is in CPU memory not GPU
            # memory before attempting to reslice it
            dict_datasets_pipeline[next_section_in] = _perform_reslice(
                dict_datasets_pipeline[next_section_in],
                section,
                platform_sections[i+1],
                reslice_info,
                comm
            )
    ##************* Sections loop is complete *************## 

    """
    # Run the methods
    for idx, method_func in enumerate(method_funcs[1:]):
        package = method_func.module_name.split(".")[0]
        method_name = method_func.parameters.pop("method_name")
        task_no_str = f"Running task {idx+2}"
        task_end_str = task_no_str.replace("Running", "Finished")
        pattern_str = f"(pattern={method_func.pattern.name})"
        log_once(
            f"{task_no_str} {pattern_str}: {method_name}...",
            comm,
            colour=Colour.LIGHT_BLUE,
            level=0,
        )
        start = time.perf_counter_ns()

        # check if the module needs the ncore parameter and add it
        if "ncore" in signature(method_func.method_func).parameters:
            method_func.parameters.update({"ncore": ncore})

        idx += 1
        run_method(
            idx,
            save_all,
            possible_extra_params,
            method_func,
            dict_datasets_pipeline,
            comm,
        )

        stop = time.perf_counter_ns()
        output_str_list = [
            f"    {task_end_str} {pattern_str}: {method_name} (",
            package,
            f") Took {float(stop-start)*1e-6:.2f}ms",
        ]
        output_colour_list = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
        log_once(output_str_list, comm=comm, colour=output_colour_list)

    reslice_summary_str = f"Total number of reslices: {reslice_info.count}"
    reslice_summary_colour = Colour.BLUE if reslice_info.count <= 1 else Colour.RED
    log_once(reslice_summary_str, comm=comm, colour=reslice_summary_colour, level=1)
    """

    elapsed_time = 0.0
    if comm.rank == 0:
        elapsed_time = MPI.Wtime() - start_time
        end_str = f"~~~ Pipeline finished ~~~ took {elapsed_time} sec to run!"
        log_once(end_str, comm=comm, colour=Colour.BVIOLET)
        #: remove ansi escape sequences from the log file
        remove_ansi_escape_sequences(f"{httomo.globals.run_out_dir}/user.log")


def _initialise_datasets_and_stats(
    yaml_config: Path,
) -> tuple[Dict[str, None], List[Dict]]:
    """Add keys to dict that will contain all datasets defined in the YAML
    config.

    Parameters
    ----------
    yaml_config : Path
        The file containing the processing pipeline info as YAML

    Returns
    -------
    tuple
        Returns a tuple containing a dict of datasets and a
        list containing the stats of all datasets of all methods in the pipeline.
        The fist element is the dict of datasets, whose keys are the names of the datasets, and
        values will eventually be arrays (but initialised to None in this
        function)
    """
    datasets, stats = {}, []
    # Define a list of parameter names that refer to a "dataset" that would need
    # to exist in the `datasets` dict
    loader_dataset_param = "name"
    loader_dataset_params = [loader_dataset_param]
    method_dataset_params = ["data_in", "data_out"]

    dataset_params = method_dataset_params + loader_dataset_params

    yaml_conf = open_yaml_config(yaml_config)
    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        method_name, method_conf = module_conf.popitem()
        # Check parameters of method if it contains any of the parameters which
        # require a dataset to be defined

        if "loaders" in module_name:
            dataset_param = loader_dataset_param
        else:
            dataset_param = "data_in"

        # Dict to hold the stats for each dataset associated with the method
        method_stats: Dict[str, List] = {}
        method_stats[method_conf[dataset_param]] = []
        stats.append(method_stats)

        for param in method_conf.keys():
            if param in dataset_params:
                if type(method_conf[param]) is list:
                    for dataset_name in method_conf[param]:
                        if dataset_name not in datasets:
                            datasets[dataset_name] = None
                else:
                    if method_conf[param] not in datasets:
                        datasets[method_conf[param]] = None

    return datasets, stats


def _get_method_funcs(yaml_config: Path, comm: MPI.Comm) -> List[MethodFunc]:
    """Gather all the python functions needed to run the defined processing
    pipeline.

    Parameters
    ==========

    yaml_config : Path
        The file containing the processing pipeline info as YAML

    Returns
    =======

    List[MethodFunc]
        A list describing each method function with its properties
    """
    method_funcs: List[MethodFunc] = []
    yaml_conf = open_yaml_config(yaml_config)
    methods_count = len(yaml_conf)

    # the first task is always the loader
    # so consider it separately
    assert next(iter(yaml_conf[0].keys())) == "httomo.data.hdf.loaders"
    module_name, module_conf = yaml_conf[0].popitem()
    method_name, method_conf = module_conf.popitem()
    method_conf["method_name"] = method_name
    module = import_module(module_name)
    method_func = getattr(module, method_name)
    method_funcs.append(
        MethodFunc(
            module_name=module_name,
            method_func=method_func,
            wrapper_func=None,
            parameters=method_conf,
            is_loader=True,
            cpu=True,
            gpu=False,
            pattern=Pattern.all,
        )
    )

    for i, task_conf in enumerate(yaml_conf[1:]):
        module_name, module_conf = task_conf.popitem()
        split_module_name = module_name.split(".")
        method_name, method_conf = module_conf.popitem()
        method_conf["method_name"] = method_name

        if split_module_name[0] not in ["tomopy", "httomolib", "httomolibgpu"]:
            err_str = (
                f"An unknown module name was encountered: " f"{split_module_name[0]}"
            )
            log_exception(err_str)
            raise ValueError(err_str)

        module_to_wrapper = {
            "tomopy": TomoPyWrapper,
            "httomolib": HttomolibWrapper,
            "httomolibgpu": HttomolibgpuWrapper,
        }
        wrapper_init_module = module_to_wrapper[split_module_name[0]](
            split_module_name[1], split_module_name[2], method_name, comm
        )
        wrapper_func = getattr(wrapper_init_module.module, method_name)
        wrapper_method = wrapper_init_module.wrapper_method
        is_tomopy = split_module_name[0] == "tomopy"
        is_httomolib = split_module_name[0] == "httomolib"
        is_httomolibgpu = split_module_name[0] == "httomolibgpu"

        method_funcs.append(
            MethodFunc(
                module_name=module_name,
                method_func=wrapper_func,
                wrapper_func=wrapper_method,
                parameters=method_conf,
                cpu=True if not is_httomolibgpu else wrapper_init_module.meta.cpu,
                gpu=False if not is_httomolibgpu else wrapper_init_module.meta.gpu,
                calc_max_slices=None
                if not is_httomolibgpu
                else wrapper_init_module.calc_max_slices,
                pattern=Pattern.all,
                is_loader=False,
                return_numpy=False,          
            )
        )

    return method_funcs


def run_method(
    task_idx: int,
    save_all: bool,
    misc_params: List[Tuple[List[str], object]],
    current_func: MethodFunc,
    dict_datasets_pipeline: Dict[str, Optional[ndarray]],
    comm: MPI.Comm,
) -> Dict:
    """
    Run a method function in the processing pipeline.

    Parameters
    ----------
    task_idx : int
        The index of the current task (zero-based indexing).
    save_all : bool
        Whether to save the result of all methods in the pipeline.
    misc_params : List[Tuple[List[str], object]]
        A list of possible extra params that may be needed by a method.
    current_func : MethodFunc
        Object describing the python function that performs the method.
    dict_datasets_pipeline : Dict[str, ndarray]
        A dict containing all available datasets in the given pipeline.
    comm : MPI.Comm
        The MPI communicator used for the run.

    Returns
    -------
    Dict
        Returns the global stats
    """
    module_path = current_func.module_name
    method_name = current_func.method_func.__name__
    func_wrapper = current_func.wrapper_func
    package_name = current_func.module_name.split(".")[0]

    #: create an object that would be passed along to prerun_method,
    #: run_method, and postrun_method
    run_method_info = RunMethodInfo(task_idx=task_idx)

    #: prerun - before running the method, update the dictionaries
    prerun_method(
        run_method_info,
        save_all,
        misc_params,
        current_func,
        dict_datasets_pipeline,
    )

    # Assign the `data` parameter of the method to the array associated with its
    # input dataset
    run_method_info.dict_httomo_params["data"] = dict_datasets_pipeline[
        run_method_info.data_in
    ]

    # Run method on the input data
    if method_name == "save_to_images":
        func_wrapper(
            method_name,
            run_method_info.dict_params_method,
            **run_method_info.dict_httomo_params,
        )
    else:
        res = func_wrapper(
            method_name,
            run_method_info.dict_params_method,
            **run_method_info.dict_httomo_params,
        )
        # Store the output(s) of the method in the appropriate
        # dataset in the `dict_datasets_pipeline` dict
        if isinstance(res, (tuple, list)):
            # The method produced multiple outputs
            for val, dataset in zip(res, run_method_info.data_out):
                dict_datasets_pipeline[dataset] = val
        else:
            # The method produced a single output
            dict_datasets_pipeline[run_method_info.data_out] = res

    if method_name == "save_to_images":
        # Nothing more to do if the saver has a special kind of
        # output which handles saving the result
        return

    postrun_method(run_method_info, dict_datasets_pipeline, current_func)


def _check_params_for_sweep(params: Dict) -> int:
    """Check the parameter dict of a method for the number of parameter sweeps
    that occur.
    """
    count = 0
    for k, v in params.items():
        if type(v) is tuple:
            count += 1
    return count


def _assign_pattern_to_method(method_function: MethodFunc) -> MethodFunc:
    """Fetch the pattern information from the methods database in
    `httomo/methods_database/packages` for the given method and associate that
    pattern with the function object.

    Parameters
    ----------
    method_function : MethodFunc
        The method function information whose pattern information will be fetched and populated.

    Returns
    -------
    MethodFunc
        The function information `pattern` attribute set, corresponding to the
        pattern that the method requires its input data to have.
    """
    pattern_str = get_method_info(
        method_function.module_name, method_function.method_func.__name__, "pattern"
    )
    if pattern_str == "projection":
        pattern = Pattern.projection
    elif pattern_str == "sinogram":
        pattern = Pattern.sinogram
    elif pattern_str == "all":
        pattern = Pattern.all
    else:
        err_str = (
            f"The pattern {pattern_str} that is listed for the method "
            f"{method_function.module_name} is invalid."
        )
        log_exception(err_str)
        raise ValueError(err_str)

    return dataclasses.replace(method_function, pattern=pattern)


def _determine_platform_sections(
    method_funcs: List[MethodFunc],
) -> List[PlatformSection]:
    ret: List[PlatformSection] = []
    current_gpu = method_funcs[0].gpu
    current_pattern = method_funcs[0].pattern
    methods: List[MethodFunc] = []
    for method in method_funcs:
        if method.gpu == current_gpu and (
            method.pattern == current_pattern
            or method.pattern == Pattern.all
            or current_pattern == Pattern.all
        ):
            methods.append(method)
            if current_pattern == Pattern.all and method.pattern != Pattern.all:
                current_pattern = method.pattern
        else:
            ret.append(
                PlatformSection(
                    gpu=current_gpu,
                    pattern=current_pattern,
                    max_slices=0,
                    methods=methods,
                    output_stats=None,
                )
            )
            methods = [method]
            current_pattern = method.pattern
            current_gpu = method.gpu

    ret.append(
        PlatformSection(
            gpu=current_gpu, pattern=current_pattern, max_slices=0, methods=methods,
            output_stats=None
        )
    )

    return ret


def _get_available_gpu_memory(safety_margin_percent: float = 10.0) -> int:
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        # first, let's make some space
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        cache = cp.fft.config.get_plan_cache()
        cache.clear()
        available_memory = dev.mem_info[0] + pool.free_bytes()
        return int(available_memory * (1 - safety_margin_percent / 100.0))
    except:
        return int(100e9)  # arbitrarily high number - only used if GPU isn't available


def _update_max_slices(
    section: PlatformSection,
    process_data_shape: Optional[Tuple[int, int, int]],
    input_data_type: Optional[np.dtype],
) -> Tuple[np.dtype, Tuple[int, int]]:
    if process_data_shape is None or input_data_type is None:
        return
    if section.pattern == Pattern.sinogram:
        slice_dim = 1
        non_slice_dims_shape = (process_data_shape[0], process_data_shape[2])
    elif section.pattern == Pattern.projection or section.pattern == Pattern.all:
        # TODO: what if all methods in a section are pattern.all
        slice_dim = 0
        non_slice_dims_shape = (process_data_shape[1], process_data_shape[2])
    else:
        err_str = f"Invalid pattern {section.pattern}"
        log_exception(err_str)
        # this should not happen if data type is indeed the enum
        raise ValueError(err_str)
    max_slices = process_data_shape[slice_dim]
    data_type = input_data_type
    output_dims = non_slice_dims_shape
    if section.gpu:
        available_memory = _get_available_gpu_memory(10.0)
        available_memory_in_GB = round(available_memory / (1024**3), 2)
        max_slices_methods = [max_slices] * len(section.methods)
        idx = 0
        for m in section.methods:
            if m.calc_max_slices is not None:
                (slices_estimated, data_type, output_dims) = m.calc_max_slices(
                    slice_dim, non_slice_dims_shape, data_type, available_memory
                )
                max_slices_methods[idx] = min(max_slices, slices_estimated)
                idx += 1
            non_slice_dims_shape = (
                output_dims  # overwrite input dims with estimated output ones
            )
        section.max_slices = min(max_slices_methods)
    else:
        # TODO: How do we determine the output dtype in functions that aren't on GPU, tomopy, etc.
        section.max_slices = max_slices
        pass
    return data_type, output_dims


def _perform_reslice(
    data: np.ndarray,
    current_section: PlatformSection,
    next_section: PlatformSection,
    reslice_info: ResliceInfo,
    comm: MPI.Comm,
) -> np.ndarray:
    reslice_info.count += 1
    if reslice_info.count > 1 and not reslice_info.has_warn_printed:
        reslice_warn_str = (
            "WARNING: Reslicing is performed more than once in this "
            "pipeline, is there a need for this?"
        )
        log_once(reslice_warn_str, comm=comm, colour=Colour.RED)
        reslice_info.has_warn_printed = True

    current_slice_dim = _get_slicing_dim(current_section.pattern)
    next_slice_dim = _get_slicing_dim(next_section.pattern)

    if reslice_info.reslice_dir is None:
        resliced_data, _ = reslice(
            data,
            current_slice_dim,
            next_slice_dim,
            comm,
        )
    else:
        resliced_data, _ = reslice_filebased(
            data,
            current_slice_dim,
            next_slice_dim,
            comm,
            reslice_info.reslice_dir,
        )

    return resliced_data
