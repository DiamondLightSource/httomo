import multiprocessing
from datetime import datetime
from os import mkdir
from pathlib import Path
from typing import List, Dict, Tuple
from collections.abc import Callable
from inspect import signature

from numpy import ndarray
from mpi4py import MPI

from httomo.yaml_utils import open_yaml_config
from httomo.data.hdf import loaders
from wrappers.tomopy import misc, prep, recon


def run_tasks(
    in_file: Path,
    yaml_config: Path,
    out_dir: Path,
    dimension: int,
    crop: int = 100,
    pad: int = 0,
    ncores: int = 1
):
    """Run the tomopy pipeline defined in the YAML config file

    Args:
        in_file: The file to read data from.
        yaml_config: The file containing the processing pipeline info as YAML
        out_dir: The directory to write data to.
        data_key: The input file dataset key to read.
        dimension: The dimension to slice in.
        crop: The percentage of data to use. Defaults to 100.
        pad: The padding size to use. Defaults to 0.
        ncores: The number of the CPU cores per process
    """
    comm = MPI.COMM_WORLD
    run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )
    if comm.rank == 0:
        mkdir(run_out_dir)
    if comm.size == 1:
        ncores = multiprocessing.cpu_count() # use all available CPU cores if not an MPI run

    # Get a list of the python functions associated to the methods defined in
    # user config YAML
    method_funcs = _get_method_funcs(yaml_config)

    # Define dict of params that are needed by loader functions
    loader_extra_params = {
        'in_file': in_file,
        'dimension': dimension,
        'crop': crop,
        'pad': pad,
        'comm': comm
    }

    # Run the methods
    for idx, (func, params, is_loader) in enumerate(method_funcs):
        print(f"Running method {idx+1}...")
        if is_loader:
            params.update(loader_extra_params)
            data, flats, darks, angles, angles_total, detector_y, detector_x = \
                _run_loader(func, params)
        else:
            method_name = params.pop('method_name')
            data_params = _check_signature_for_data_params(func, darks, flats,
                                                           angles)
            #dataset_params = {
            #    'data_in': params.pop('data_in'),
            #    'data_out': params.pop('data_out')
            #}

            # Add `data` as another param
            data_params.update({'data': data})
            data = _run_method(func, method_name, params, data_params)


def _get_method_funcs(yaml_config: Path) -> List[Tuple[Callable, Dict, bool]]:
    """Gather all the python functions needed to run the defined processing
    pipeline

    Args:
        yaml_config: The file containing the processing pipeline info as YAML
    """
    method_funcs = []
    yaml_conf = open_yaml_config(yaml_config)

    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        split_module_name = module_name.split('.')
        if split_module_name[0] == 'httomo':
            if 'loaders' in module_name:
                is_loader = True
            else:
                is_loader = False
            module_name = split_module_name[-1]
            module = globals()[module_name]
            method_name, method_conf = module_conf.popitem()
            method_func = getattr(module, method_name)
            method_funcs.append((method_func, method_conf, is_loader))
        elif split_module_name[0] == 'tomopy':
            # The structure of wrapper functions for tomopy is that each module
            # in tomopy is represented by a function in HTTomo.
            #
            # For example, the module `tomopy.misc.corr` module (which contains
            # the `median_filter()` method function) is represented in HTTomo in
            # the `wrappers.tomopy.misc` module as the `corr()` function.
            #
            # Different method functions that are available in the
            # `tomopy.misc.corr` module are then exposed in HTTomo by passing a
            # `method_name` parameter to the corr() function via the YAML config
            # file.
            wrapper_module_name = split_module_name[-2]
            wrapper_module = globals()[wrapper_module_name]
            wrapper_func_name = split_module_name[-1]
            wrapper_func = getattr(wrapper_module, wrapper_func_name)
            method_name, method_conf = module_conf.popitem()
            method_conf['method_name'] = method_name
            method_funcs.append((wrapper_func, method_conf, False))
        else:
            err_str = f"An unknown module name was encountered: " \
                      f"{split_module_name[0]}"
            raise ValueError(err_str)

    return method_funcs


def _run_loader(func: Callable, params: Dict) -> Tuple[ndarray, ndarray,
                                                       ndarray, ndarray,
                                                       ndarray, int, int, int]:
    """Run a loader function in the processing pipeline.

    Args:
        func: The python function that performs the loading.
        params: A dict of parameters for the loader.
    """
    return func(**params)


def _run_method(func: Callable, method_name:str, method_params: Dict,
                data_params: Dict) -> None:
    """Run a method function in the processing pipeline.

    Args:
        func: The python function that performs the method.
        method_name: The name of the method to apply.
        method_params: A dict of parameters for the method.
        data_params: A dict of parameters related to the data from a loader.
    """
    return func(method_params, method_name, **data_params)


def _check_signature_for_data_params(func: Callable, darks: ndarray,
                                     flats: ndarray, angles: ndarray) -> Dict:
    """Check if the given method requires any parameters related to the loader.

    Args:
        func: The method function to inspect.
        darks: The darks to potentially pass onto the method function.
        darks: The flats to potentially pass onto the method function.
        darks: The angles to potentially pass onto the method function.

    Returns:
        Dict: The updates parameter dict for the method function
    """
    data_params = {}
    func_params = signature(func).parameters
    if 'darks' in func_params:
        data_params['darks'] = darks
    if 'flats' in func_params:
        data_params['flats'] = flats
    if 'angles' in func_params:
        data_params['angles'] = angles

    return data_params
