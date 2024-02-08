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
from typing import TypeAlias, Dict, List, Optional, Tuple

import numpy as np
from httomolib.misc.images import save_to_images
from mpi4py import MPI
from numpy import ndarray

from httomo.utils import gpu_enabled, xp
from httomo.data.mpiutil import local_rank

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
from httomo.wrappers_class import BackendWrapper, _gpumem_cleanup
from httomo.yaml_utils import open_yaml_config
supporting_module_path = "httomo.methods_database.packages.external."


SliceIndices: TypeAlias = List[Tuple[int, int, int]]


def run_tasks(
    in_file: Path,
    yaml_config: Path,
    save_all: bool = False,
    reslice_dir: Optional[Path] = None,
) -> None:
    """Run the pipeline defined in the YAML config file

    Parameters
    ----------
    in_file
        The file to read data from.
    yaml_config
        The file containing the processing pipeline info as YAML.
    save_all
        Specifies if intermediate datasets should be saved for all tasks in the
        pipeline.
    reslice_dir
        Path where to store the reslice intermediate files, or None if reslicing
        should be done in-memory.
    """
    comm = MPI.COMM_WORLD

    # Define dict to store arrays of the whole pipeline using provided YAML
    # Define list to store dataset stats for each task in the user config YAML
    dict_datasets_pipeline, glob_stats = _initialise_datasets_and_stats(yaml_config)

    # Get a list of the python functions associated to the methods defined in
    # user config YAML
    method_funcs = _get_method_funcs(yaml_config, comm)

    # Associate patterns to method function objects
    for i, method_func in enumerate(method_funcs):
        method_funcs[i] = _assign_attribute_to_method(method_func, "pattern")        

    # Associate output_dims_change method attribute to method function objects
    for i, method_func in enumerate(method_funcs):
         method_funcs[i] = _assign_attribute_to_method(method_func, "output_dims_change")

    # Associate implementation architectures to method function objects
    for i, method_func in enumerate(method_funcs):
         method_funcs[i] = _assign_attribute_to_method(method_func, "implementation")

    # Associate memory attributes to method function objects
    for i, method_func in enumerate(method_funcs):
         method_funcs[i] = _assign_attribute_to_method(method_func, "memory_gpu")

    # Associate padding attribute to method function objects
    for i, method_func in enumerate(method_funcs):
         method_funcs[i] = _assign_attribute_to_method(method_func, "padding")

    # initiate glob_stats in methods so that division into sections would be correct
    _initiate_stats_for_methods(method_funcs[1:])
    
    # Initialising platform sections (skipping loader method)
    platform_sections = _determine_platform_sections(method_funcs[1:], save_all)        
    
    _evaluate_padding_per_section(platform_sections)

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
    
    # padding value in the first section will define padding for the loader
    padding_to_loader = platform_sections[0].padding
    # Define dict of params that are needed by loader functions
    dict_loader_extra_params = {
        "in_file": in_file,
        "pad": padding_to_loader,
        "comm": comm,
    }

    # store info about reslicing with ResliceInfo
    reslice_info = ResliceInfo(
        count=0, has_warn_printed=False, reslice_dir=reslice_dir
    )
    method_funcs[0].parameters.update(dict_loader_extra_params)

    # Check if a value for the `preview` parameter of the loader has
    # been provided
    if "preview" not in method_funcs[0].parameters.keys():
        method_funcs[0].parameters["preview"] = [None]
    
    output_colour_list = [Colour.GREEN, Colour.CYAN, Colour.GREEN]
    output_colour_list_short = [Colour.GREEN, Colour.CYAN]
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
    log_once(output_str_list, comm=comm, colour=output_colour_list)

    # Update `dict_datasets_pipeline` dict with the data that has been
    # loaded by the loader
    # NOTE: conversion to float32 to avoid creating an additional output array in the GPU loop
    dict_datasets_pipeline[method_funcs[0].parameters["name"]] = np.float32(loader_info.data)
    dict_datasets_pipeline["flats"] = np.float32(loader_info.flats)
    dict_datasets_pipeline["darks"] = np.float32(loader_info.darks)
    # Clean up `loader_info.data`, `loader_info.darks`, and `loader_info.flats`,
    # as they have been converted to float32 and the original uint16 versions
    # are no longer needed
    del loader_info.data
    del loader_info.darks
    del loader_info.flats

    # Extra params relevant to httomo that a wrapper function might need
    possible_extra_params = [
        (["darks"], dict_datasets_pipeline["darks"]),
        (["flats"], dict_datasets_pipeline["flats"]),
        (["angles", "angles_radians"], loader_info.angles),
        (["comm"], comm),
        (["out_dir"], httomo.globals.run_out_dir),
        (["return_numpy"], False),
        (["cupyrun"], False),
    ]
    # data shape and dtype are useful when calculating max slices
    data_shape = dict_datasets_pipeline[method_funcs[0].parameters["name"]].shape
    # data_dtype = loader_info.data.dtype
    data_dtype = np.dtype(np.float32) # make the data type constant for the run
    
    ##---------- MAIN LOOP STARTS HERE ------------##
    idx = 0
    # initialise the CPU data array with the loaded data, we override it at the end of every section
    data_full_section = dict_datasets_pipeline[method_funcs[idx].parameters["name"]]
    for i, section in enumerate(platform_sections):  
        # getting the slicing dimension of the section
        slicing_dim_section = _get_slicing_dim(section.pattern) - 1        
        
        # determine max_slices for the whole section and return output dims and type
        output_dims_upd, data_type_upd = _update_max_slices(section,
                                                            slicing_dim_section,
                                                            data_shape,
                                                            data_dtype,
                                                            dict_datasets_pipeline)
        
        maxslices_str = f"Maximum amount of slices is {section.max_slices} for section {i}"
        log_once(maxslices_str, comm=comm, colour=Colour.BVIOLET, level=1)

        # iterations_for_blocks determines the number of iterations needed
        # to raster through the data in blocks that would fit into the GPU memory        
        if section.max_slices == data_shape[slicing_dim_section]:
            # in this case the whole data CHUNK fits into a BLOCKб so there is no need to pad
            section.padding = 0        
        section.max_slices -= 2*section.padding # reduce the amount of slices to account for padding
        iterations_for_blocks = math.ceil(data_shape[slicing_dim_section] / section.max_slices)
        # Define a list to store the `RunMethodInfo` objects created for each
        # method in the methods-loop, for reuse across iterations over blocks
        run_method_info_objs = []
        # Check if any methods in the section are a recon method, because recon
        # methods change the shape of the input data and thus the processed
        # blocks cannot be straightforwardly put into `data_full_section`.
        # Instead, a new numpy array is created with the necessary shape to hold
        # the full reconstructed volume, and it's this array that is updated as
        # blocks are processed in the blocks-loop
        contains_recon = any(['recon.algorithm' in method_func.module_name for method_func in section.methods])
        if contains_recon:
            recon_arr = np.empty(output_dims_upd, dtype=np.float32)
       
        ##---------- LOOP OVER _BLOCKS_ IN THE SECTION ------------##
        indices_start = 0
        indices_end = int(section.max_slices)
        slc_indices = [slice(None)] * len(data_shape)
        for it_blocks in range(iterations_for_blocks):
            # Preparing indices for the slicing of the data in blocks
            slc_indices[slicing_dim_section] = _get_slc_indices(
                section, data_shape, slicing_dim_section, indices_start, indices_end
            )

            ##---------- LOOP OVER _METHODS_ IN THE BLOCK ------------##
            for m_ind, methodfunc_sect in enumerate(section.methods):
                # preparing everything for the wrapper execution
                module_path = methodfunc_sect.module_name
                method_name = methodfunc_sect.method_func.__name__
                func_wrapper = methodfunc_sect.wrapper_func
                package_name = methodfunc_sect.module_name.split(".")[0]
                
                # log related stuff
                pattern_str = f"(pattern={section.pattern.name})"
                package_str = f"({package_name})"
                task_no_str = f"Running task {idx+2}"
                task_end_str = task_no_str.replace("Running", "Finished")
               
                if it_blocks == iterations_for_blocks-1:
                    log_once(
                    f"{task_no_str} {pattern_str}: {method_name}...",
                    comm,
                    colour=Colour.LIGHT_BLUE,
                    level=0,
                    )
                # Only run the `prerun_method()` once for a method, when the
                # first block is going to be processed. This is because the
                # method's parameters will not have changed from the first time
                # it has run, only its input data, which is taken care of
                # outside the `prerun_method()` function.
                if it_blocks == 0:
                    #: create an object that would be passed along to prerun_method,
                    #: run_method, and postrun_method
                    run_method_info = RunMethodInfo(task_idx=m_ind)
                    run_method_info_objs.append(run_method_info)

                    #: prerun - before running the method, update the dictionaries
                    prerun_method(
                        run_method_info,
                        section,
                        possible_extra_params,
                        methodfunc_sect,
                        dict_datasets_pipeline,
                        save_all,
                    )

                    # remove methods name from the parameters list of a method
                    #
                    # TODO: As this value is not being used anywhere, and the
                    # `method_name` variable is being defined by other means,
                    # the addition of the method name to the dict of method
                    # parameters in `_get_method_funcs()` possibly could be
                    # removed?
                    run_method_info.dict_params_method.pop('method_name')
                else:
                    # Reuse the `RunMethodInfo` object that was defined for the
                    # method when the 0th block in the section was processed
                    run_method_info = run_method_info_objs[m_ind]

                if m_ind == 0:
                    # Assign the block of data on a CPU to `data` parameter of
                    # the method; this should happen once in the beginning of
                    # the loop over methods
                    run_method_info.dict_httomo_params["data"] = data_full_section[
                        tuple(slc_indices)
                    ]
                else:
                    # Initialise with result from the previous method                    
                    run_method_info.dict_httomo_params["data"] = res                    
                
                # Override the `data` param in the special case of a centering
                # method
                if 'center' in method_name and it_blocks == 0:
                    slice_ind_center = run_method_info.dict_params_method['ind']
                    if slice_ind_center is None or 'mid':
                        slice_ind_center = data_full_section.shape[1] // 2  # get the middle slice of the whole data chunk
                    # copy the "ind" slice from the last section dataset (a chunk)
                    # NOTE: even if there are some GPU filters before in the section, 
                    # we still be using the data from the LAST section                    
                    run_method_info.dict_httomo_params["data"] = data_full_section[:,slice_ind_center,:]
  
                # Calculate stats on the result of the last method (except centering)
                # It is triggered by adding glob_stats: true parameter to the parameters list
                if 'center' not in method_name and i < len(platform_sections) and run_method_info.global_statistics:
                    run_method_info.dict_params_method['glob_stats'] = min_max_mean_std(data_full_section, comm)

                # ------ RUNNING THE WRAPPER -------#
                start = time.perf_counter_ns()
                if 'center' in method_name:
                    if it_blocks == 0:
                        # we need to avoid overriding "res" with a scalar.                        
                        res_param = func_wrapper(
                            method_name,
                            run_method_info.dict_params_method,
                            **run_method_info.dict_httomo_params,
                        )
                        # Store the output(s) of the method in the appropriate
                        # dataset in the `dict_datasets_pipeline` dict
                        if isinstance(res_param, (tuple, list)):
                            # The method produced multiple outputs
                            for val, dataset in zip(res_param, run_method_info.data_out):
                                dict_datasets_pipeline[dataset] = val
                        else:
                            # The method produced a single output
                            dict_datasets_pipeline[run_method_info.data_out] = res_param                       
                    # passing the data to the next method (or initialise res)
                    if m_ind == 0:
                        res = data_full_section[tuple(slc_indices)]
                elif method_name == "save_to_images":
                    # just executing the wrapper (no output)
                    func_wrapper(
                            method_name,
                            run_method_info.dict_params_method,
                            **run_method_info.dict_httomo_params,
                        )
                    # passing the data to the next method (or initialise res)
                    if m_ind == 0:
                        res = data_full_section[tuple(slc_indices)]
                else:
                    # overriding the result with an output of a method
                    res = func_wrapper(
                        method_name,
                        run_method_info.dict_params_method,
                        **run_method_info.dict_httomo_params,
                    )
                # ------ WRAPPER COMPLETED -------#
                stop = time.perf_counter_ns()

                section_block_method_str = f"Section {i} runs method {method_name} on a block {it_blocks} of {indices_end-indices_start} slices"
                output_str_list_verbose = [
                    f"{section_block_method_str} ",
                    f" Complete in {float(stop-start)*1e-6:.2f}ms",
                ]
                log_once(output_str_list_verbose, comm=comm, colour=output_colour_list_short, level = 1)
               
                if it_blocks == iterations_for_blocks-1:
                    output_str_list_once = [
                        f"    {task_end_str} {pattern_str}: {method_name} ",
                        f" {package_str}",
                    ]              
                    log_once(output_str_list_once, comm=comm, colour=output_colour_list_short)
                    idx += 1
                
                if isinstance(res, (tuple, list)):
                    err_str = (
                        "Methods producing multiple outputs are not yet "
                        "supported in the GPU loop"
                    )
                    raise ValueError(err_str)
    
            ##************* METHODS LOOP IS COMPLETE *************##            
            # cropping indices for a block if the block-data is padded
            slc_block_cropping_indices = _get_crop_indices(
                section,
                data_shape,
                slicing_dim_section,
                res,
                indices_start,
                indices_end,
            )
            # Saving the processed block (the block is in the CPU memory)
            if not contains_recon:
                data_full_section[tuple(slc_indices)] = res[tuple(slc_block_cropping_indices)]
            else:
                recon_arr[tuple(slc_indices)] = res[tuple(slc_block_cropping_indices)]

            # re-initialise the slicing indices
            indices_start = indices_end
            # checking if we still within the slicing dimension size and take remaining portion
            res_indices = (indices_start + int(section.max_slices)) - data_shape[slicing_dim_section]
            if res_indices > 0:
                res_indices = int(section.max_slices - res_indices)
                indices_end += res_indices
            else:
                indices_end += section.max_slices
            
            # delete allocated memory pointers to free up the memory
            run_method_info.dict_httomo_params["data"] = None
            # now flushing the GPU memory
            _gpumem_cleanup()
            
        ##************* BLOCKS LOOP IS COMPLETE *************##
        # If the completed section contained a recon method, then the array
        # created to hold the differently-shaped output of this section (due to
        # the recon within the section changing the shape of the input data)
        # must be assigned to
        # - the original dataset name (defined by the loader) in
        # `dict_datasets_pipeline`
        # - the `data_full_section` variable
        if contains_recon:
            dict_datasets_pipeline[method_funcs[0].parameters["name"]] = \
                recon_arr
            data_full_section = recon_arr
        
        # saving intermediate datasets IF it has been asked for
        postrun_method(
            run_method_info,
            dict_datasets_pipeline,
            section,
            loader_info
        )

        if section.reslice:
            # we reslice only when the pattern of the section changes
            next_section_in = platform_sections[i+1].methods[0].parameters["data_in"]
            dict_datasets_pipeline[next_section_in] = _perform_reslice(
                dict_datasets_pipeline[next_section_in],
                section,
                platform_sections[i+1],
                reslice_info,
                comm
            )
            # re-initialise the section input with the resliced data
            data_full_section = dict_datasets_pipeline[next_section_in]
        
        # update input data dimensions and data type for the next section        
        data_shape = np.shape(data_full_section)
        data_dtype = data_type_upd

    ##************* SECTIONS LOOP IS COMPLETE *************##
    elapsed_time = 0.0
    if comm.rank == 0:
        elapsed_time = MPI.Wtime() - start_time
        end_str = f"~~~ Pipeline finished ~~~ took {elapsed_time} sec to run!"
        log_once(end_str, comm=comm, colour=Colour.BVIOLET)
        #: remove ansi escape sequences from the log file
        remove_ansi_escape_sequences(f"{httomo.globals.run_out_dir}/user.log")


def _get_slc_indices(
    section: PlatformSection,
    data_shape: Tuple,
    slicing_dim_section: int,
    indices_start: int,
    indices_end: int,
) -> SliceIndices:
    """ Return slice boundaries for slicing data

    Parameters
    ----------
    section
        Section of data.
    data_shape
    slicing_dim_section
        Dimension in which to slice.
    indices_start
        Start indices for the slicing of the data.
    indices_end
        End indices for the slicing of the data.

    Returns
    -------
    SliceIndices
        List of indices tuples
    """
    indices_start_pad = indices_start
    indices_end_pad = indices_end
    if section.padding > 0:
        # We also need to take padding into the account here
        if (indices_start - section.padding) >= 0:
            indices_start_pad = indices_start - section.padding
        if (indices_end + section.padding) < data_shape[slicing_dim_section]:
            indices_end_pad = indices_end + section.padding
    return slice(int(indices_start_pad), int(indices_end_pad), 1)


def _get_crop_indices(
    section: PlatformSection,
    data_shape: Tuple,
    slicing_dim_section: int,
    res: int,
    indices_start: int,
    indices_end: int,
) -> SliceIndices:
    """Return cropping boundary

    Parameters
    ----------
    section
        Section of data.
    data_shape
    slicing_dim_section
        Dimension in which to slice.
    res
        Result
    indices_start
        Start indices for the slicing of the data.
    indices_end
        End indices for the slicing of the data.

    Returns
    --------
    SliceIndices
        List of indices tuples.

    """
    slc_block_cropping_indices = [slice(None)] * len(res.shape)
    if indices_start != 0:
        indices_block_start_crop = section.padding
    else:
        indices_block_start_crop = 0
    if indices_end != data_shape[slicing_dim_section]:
        indices_block_end_crop = res.shape[slicing_dim_section] - section.padding
    else:
        indices_block_end_crop = res.shape[slicing_dim_section]
    slc_block_cropping_indices[slicing_dim_section] = slice(
        int(indices_block_start_crop), int(indices_block_end_crop), 1
    )
    return slc_block_cropping_indices


def _initialise_datasets_and_stats(
    yaml_config: Path,
) -> tuple[Dict[str, None], List[Dict]]:
    """Add keys to dict that will contain all datasets defined in the YAML
    config.

    Parameters
    ----------
    yaml_config
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
    ----------
    yaml_config
        The file containing the processing pipeline info as YAML

    Returns
    -------
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

        wrapper_init_module = BackendWrapper(
            split_module_name[0],
            split_module_name[1],
            split_module_name[2],
            method_name,
            comm
        )
        wrapper_func = getattr(wrapper_init_module.module, method_name)
        wrapper_method = wrapper_init_module.wrapper_method    
        
        method_funcs.append(
            MethodFunc(
                module_name=module_name,
                method_func=wrapper_func,
                wrapper_func=wrapper_method,
                parameters=method_conf,
                cpu=True,
                gpu=False,
                cupyrun=False,
                calc_max_slices=None,
                padding = 0,
                output_dims_change = False,
                pattern=Pattern.all,
                is_loader=False,
                return_numpy=False,
                idx_global=i+2,
                global_statistics=False,
            )
        )
    return method_funcs

def _check_params_for_sweep(params: Dict) -> int:
    """Check the parameter dict of a method for the number of parameter sweeps
    that occur.
    """
    count = 0
    for k, v in params.items():
        if type(v) is tuple:
            count += 1
    return count


def _assign_attribute_to_method(
    method_function: MethodFunc,
    attribute: str,
    ) -> MethodFunc:
    """Fetch the information about a given attribute from the methods database in
    `httomo/methods_database/packages` for the given method and associate that
    with the function object.

    Parameters
    ----------
    method_function : MethodFunc
        The method function to populate.
    attribute: str
        A given attribute of a method, e.g. "pattern", "implementation", etc. 
    Returns
    -------
    MethodFunc
        The updated function object
    """
    if attribute in ["pattern", "implementation", "memory_gpu", "output_dims_change", "padding"]:
        attribute_value = get_method_info(
            method_function.module_name, method_function.method_func.__name__, attribute)
        if attribute == "pattern":
            if attribute_value in ["projection", "sinogram", "all"]:  
                if attribute_value == "projection":
                    pattern_upd = Pattern.projection
                if attribute_value == "sinogram":
                    pattern_upd = Pattern.sinogram
                if attribute_value == "all":
                    pattern_upd = Pattern.all
                method_function = dataclasses.replace(method_function, pattern=pattern_upd)
            else:
                err_str = (
                    f"The pattern {attribute_value} that is listed for the method "
                    f"{method_function.module_name} is invalid."
                )
                log_exception(err_str)
                raise ValueError(err_str)
        if attribute == "implementation":
            if attribute_value in ["cpu", "gpu", "gpu_cupy"]:
                # as "cpu" is switched by default we need to consider GPU cases here only                
                if attribute_value in ["gpu", "gpu_cupy"]:
                    method_function = dataclasses.replace(method_function, cpu=False)
                    method_function = dataclasses.replace(method_function, gpu=True)
                    if attribute_value == "gpu_cupy":
                        method_function = dataclasses.replace(method_function, cupyrun=True)
            else:
                err_str = (
                    f"The implementation arch {attribute_value} that is listed for the method "
                    f"{method_function.module_name} is invalid."
                )
                log_exception(err_str)
                raise ValueError(err_str)
        if attribute == "memory_gpu":
            if attribute_value != 'None':
                # memory gpu is active and needs to be calculated, saving params
                method_function = dataclasses.replace(method_function, calc_max_slices=attribute_value)
        if attribute == "output_dims_change":
            if attribute_value:
                method_function = dataclasses.replace(method_function, output_dims_change=True)
        if attribute == "padding":
            method_function = dataclasses.replace(method_function, padding=attribute_value)                
    else:
        err_str = (
            f"The attribute {attribute} that is listed for the method "
            f"{method_function.module_name} is invalid."
        )
        log_exception(err_str)
        raise ValueError(err_str)
    
    return method_function

def _initiate_stats_for_methods(
    method_funcs: List[MethodFunc],
) -> List[MethodFunc]:
    for m_ind, method in enumerate(method_funcs):
        try:
            global_stats = method.parameters['glob_stats']
        except:
            global_stats = False
        if global_stats:
            # the stats has been requested, the section needs to be created
            # between the current AND the previous method
            if m_ind >= 0:
                method_funcs[m_ind-1].global_statistics = True            
    return method_funcs

def _determine_platform_sections(
    method_funcs: List[MethodFunc],
    save_all: bool,
) -> List[PlatformSection]:
    section: List[PlatformSection] = []
    current_gpu = method_funcs[0].gpu
    current_pattern = method_funcs[0].pattern
    methods: List[MethodFunc] = []
    
    save_res_previous_method = save_all
    for m_ind, method in enumerate(method_funcs):
        if not save_all:
            try:
                save_res_current = method.parameters['save_result']
            except:
                save_res_current = False            
            if method.global_statistics:
                save_res_current = True # the stats has been requested, the section created
        else:
            save_res_current = True
                    
        if m_ind > 0 and save_res_previous_method:
            # previous method requested the result to be saved,
            # i.e., we need to create a section with that method or methods        
            section.append(
                PlatformSection(
                    gpu=current_gpu,
                    pattern=current_pattern,
                    reslice=False,
                    max_slices=0,
                    padding=0,
                    methods=methods,
                )
            )
            methods = [method]
            current_pattern = method.pattern
            current_gpu = method.gpu
        else:        
            if method.gpu == current_gpu and (
                method.pattern == current_pattern
                or method.pattern == Pattern.all
                or current_pattern == Pattern.all
            ):
                methods.append(method)
                if current_pattern == Pattern.all and method.pattern != Pattern.all:
                    current_pattern = method.pattern
            else:
                section.append(
                    PlatformSection(
                        gpu=current_gpu,
                        pattern=current_pattern,
                        reslice=False,
                        max_slices=0,
                        padding=0,
                        methods=methods,
                    )
                )
                methods = [method]
                current_pattern = method.pattern
                current_gpu = method.gpu
        save_res_previous_method = save_res_current

    section.append(
        PlatformSection(
            gpu=current_gpu, 
            pattern=current_pattern, 
            reslice=False, 
            max_slices=0,
            padding=0,
            methods=methods,
        )
    )
    for i, section_current in enumerate(section):
        for m_ind, methodfunc_sect in enumerate(section_current.methods):
            if m_ind == len(section_current.methods) - 1:
                # we make every last method in the section to return_numpy: True
                methodfunc_sect.return_numpy = True
            if 'rotation' in methodfunc_sect.module_name:
                # NOTE: we also need to check if centering is the last method in the section.
                # Then what architecture of the previous method in that section AND the next 
                # method in the next section after centering.
                # This is due to TomoPy CPU functions can follow centering executed on the 
                # GPU while the GPU data is not returned to CPU. 
                # Centering doesn't care about the data being in blocks as it takes data from the CPU array                
                if m_ind > 0:
                    previous_method_arch = section_current.methods[m_ind-1].gpu
                    if i > 0 and i < len(section) - 1:
                        next_method_arch = section[i+1].methods[0].gpu
                        if previous_method_arch != next_method_arch:
                            section_current.methods[m_ind-1].return_numpy = True                      
            
    # we need to check if the reslice needed _after_ the section is complete
    for i, section_current in enumerate(section):
        # and we don't need to reslice for the last section
        if i < len(section) - 1:   
            if section_current.pattern.name != section[i+1].pattern.name and section[i+1].pattern.name != 'all':
                # check that the pattern changed but exclude the case when next pattern is "all"
                section[i].reslice = True
            if i == 0 and section_current.pattern.name == 'all':
                # if the first section has "all" pattern we skip the reslice 
                section[i].reslice = False
        # exclude sections with pattern "all"
        if section_current.pattern.name == 'all':
            # inherit pattern from the previous section
            if i > 0:
                section[i].pattern = section[i-1].pattern    
    return section


def _get_available_gpu_memory(safety_margin_percent: float = 10.0) -> int:
    try:
        import cupy as cp

        dev = cp.cuda.Device(local_rank)
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
    slicing_dim_section: int,
    process_data_shape: Optional[Tuple[int, int, int]],
    input_data_type: Optional[np.dtype],
    dict_datasets_pipeline: Dict[str, ndarray],
) -> Tuple[np.dtype, Tuple[int, int]]:
    
    comm = MPI.COMM_WORLD
        
    if process_data_shape is None or input_data_type is None:
        return
    
    
    nsl_dim_l = list(process_data_shape)
    nsl_dim_l.pop(slicing_dim_section)
    non_slice_dims_shape = tuple(nsl_dim_l)
    max_slices = process_data_shape[slicing_dim_section]
    data_type = input_data_type # initialisation of data_type
    output_dims = non_slice_dims_shape # initialisation of dims
    if section.gpu:
        available_memory = int(_get_available_gpu_memory(10.0))
        available_memory_in_GB = round(available_memory / (1024**3), 2)
        memory_str = f"The amount of the available GPU memory is {available_memory_in_GB} GB"
        log_once(memory_str, comm=comm, colour=Colour.BVIOLET, level=1)
        max_slices_methods = [max_slices] * len(section.methods)
    
    idx = 0
    # loop over all methods in section
    for m in section.methods:
        if m.output_dims_change:
            # the "non_slice_dims_shape" needs to be changed for the current method
            module_attr = __library_supporting_func_getattr(m, "calc_output_dim")
            output_dims = module_attr(non_slice_dims_shape, **m.parameters)        
        if section.gpu:
            if m.calc_max_slices != 'None':
                # the gpu memory needs to be estimated for the given method
                memory_bytes_method = 1
                for i, dst in enumerate(m.calc_max_slices[0]['datasets']):
                    subtract_bytes = 0
                    # loop over the dataset names given in the library file and extracting 
                    # the corresponding dimensions from the available datasets                    
                    if dst in ["flats", "darks"]:
                        # for normalisation module dealing with flats and darks separately
                        df_bytes = np.prod(np.shape(dict_datasets_pipeline[dst])) * data_type.itemsize
                        available_memory -= int(m.calc_max_slices[1]['multipliers'][i] * df_bytes)
                    else:
                        # deal with the rest of the data                        
                        if m.calc_max_slices[2]['methods'][i] == 'direct':
                            # this calculation assumes a direct (simple) correspondence through multiplier
                            memory_bytes_method += int(m.calc_max_slices[1]['multipliers'][i] * np.prod(non_slice_dims_shape) * data_type.itemsize)
                        else:
                            # this requires more complicated memory calculation using the function that exists in methods_database of httomo
                            # building up path to the function here:
                            module_attr = __library_supporting_func_getattr(m, "calc_memory_bytes")
                            (memory_bytes_method, subtract_bytes) = module_attr(non_slice_dims_shape, data_type, **m.parameters)

                slices_estimated = (available_memory - subtract_bytes) // memory_bytes_method
            else:
                slices_estimated = max_slices            
            max_slices_methods[idx] = min(max_slices, slices_estimated)            
            idx += 1
        non_slice_dims_shape = output_dims # overwrite input dims with estimated output ones
                
    if section.gpu:
        section.max_slices = min(max_slices_methods)
    else:
        # non - gpu modules
        section.max_slices = max_slices
    # we need to return the full output_dims for the sections loop
    output_dims_l = list(output_dims)
    output_dims_l.insert(slicing_dim_section, max_slices)
    output_dims = tuple(output_dims_l)
    return output_dims, data_type

def __library_supporting_func_getattr(method, supporting_func_str):
    module_path = supporting_module_path
    for n_ind, module_str in enumerate(method.module_name.split(".")):
        if n_ind == 0:
            module_path += method.module_name.split(".")[0] + ".supporting_funcs"
        else:
            module_path += "."
            module_path += method.module_name.split(".")[n_ind]
    func_full_path = "_" + supporting_func_str + "_" + method.parameters['method_name']
    return getattr(import_module(module_path), func_full_path)


def _evaluate_padding_per_section(
    section: PlatformSection,
) -> List[PlatformSection]:
    # perform loop over sections
    for i, section in enumerate(section):
        # loop over all methods in section
        padding_methods = [0] * len(section.methods)
        for j, m in enumerate(section.methods):
            if m.padding:
                # perform estimation of the padding value using the supporting functions library
                module_attr = __library_supporting_func_getattr(m, "calc_padding")
                padding_methods[j] = module_attr(**m.parameters)
            else:
                # no padding
                padding_methods[j] = 0
        section.padding = max(padding_methods)
    return section


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
            f"WARNING: Reslicing is performed {reslice_info.count} times. The number of reslices increases the total run time."
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
