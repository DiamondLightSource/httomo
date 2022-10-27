import multiprocessing
from datetime import datetime
from os import mkdir
from pathlib import Path
from typing import List, Dict, Tuple
from collections.abc import Callable
from inspect import signature
from importlib import import_module

from numpy import ndarray
from mpi4py import MPI

from httomo.yaml_utils import open_yaml_config
from httomo.data.hdf._utils.save import intermediate_dataset


def run_tasks(
    in_file: Path,
    yaml_config: Path,
    out_dir: Path,
    dimension: int,
    crop: int = 100,
    pad: int = 0,
    ncores: int = 1,
    save_all: bool = False
) -> None:
    """Run the tomopy pipeline defined in the YAML config file

    Parameters
    ----------
    in_file : Path
        The file to read data from.
    yaml_config : Path
        The file containing the processing pipeline info as YAML.
    out_dir : Path
        The directory to write data to.
    dimension : int
        The dimension to slice in.
    crop : int
        The percentage of data to use. Defaults to 100.
    pad : int
        The padding size to use. Defaults to 0.
    ncores : int
        The number of the CPU cores per process.
    save_all : bool
        Specifies if intermediate datasets should be saved for all tasks in the
        pipeline.
    """
    comm = MPI.COMM_WORLD
    run_out_dir = out_dir.joinpath(
        f"{datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}_output"
    )
    if comm.rank == 0:
        mkdir(run_out_dir)
    if comm.size == 1:
        ncores = multiprocessing.cpu_count() # use all available CPU cores if not an MPI run

    # Define dict to store arrays that are specified as datasets in the user
    # config YAML
    datasets = _initialise_datasets(yaml_config)

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
    for idx, (package, func, params, is_loader) in enumerate(method_funcs):
        print(f"Running method {idx+1}...")
        method_name = params.pop('method_name')
        if is_loader:
            params.update(loader_extra_params)
            data, flats, darks, angles, angles_total, detector_y, detector_x = \
                _run_loader(func, params)

            # Update `datasets` dict with the data that has been loaded by the
            # loader
            datasets[params['name']] = data

            # Define all params relevant to HTTomo that a wrapper function might
            # need
            possible_extra_params = [
                (['darks'], darks),
                (['flats'], flats),
                (['angles', 'angles_radians'], angles),
                (['comm'], comm)
            ]
        else:
            save_result = False

            # Default behaviour for saving datasets is to save the output of the
            # last task.
            #
            # The default behaviour can be overridden in two ways:
            # 1. the flag `--save_all` which affects all tasks
            # 2. the method param `save_result` which affects individual tasks
            #
            # Here, enforce default behaviour.
            if idx == len(method_funcs) - 1:
                save_result = True

            # Now, check if `--save_all` has been specified, as this can
            # override default behaviour
            if save_all:
                save_result = True

            # Finally, check if `save_result` param has been specified in the
            # YAML config, as it can override both default behaviour and the
            # `--save_all` flag
            if 'save_result' in params.keys():
                save_result = params.pop('save_result')

            # Check for any extra params unrelated to tomopy but related to
            # HTTomo that should be added in
            httomo_params = \
                _check_signature_for_httomo_params(func, possible_extra_params)

            data_in = params.pop('data_in')
            data_out = params.pop('data_out')

            # Add the appropriate dataset as the `data` parameter to the method
            # function's dict of parameters
            httomo_params['data'] = datasets[data_in]

            # Check if the method function's params require any datasets stored
            # in the `datasets` dict
            dataset_params = _check_method_params_for_datasets(params, datasets)
            # Update the relevant parameter values according to the required
            # datasets
            params.update(dataset_params)

            # Run the method, then store the result in the appropriate dataset
            # in the `datasets` dict
            datasets[data_out] = \
                _run_method(func, method_name, params, httomo_params)

            # TODO: The dataset saving functionality only supports 3D data
            # currently, so check that the dimension of the data is 3 before
            # saving it
            is_3d = len(datasets[data_out].shape) == 3
            # Save the result if necessary
            if save_result and is_3d:
                intermediate_dataset(datasets[data_out], run_out_dir,
                                     comm, idx+1, package, method_name,
                                     recon_algorithm=params.pop('algorithm', None))


def _initialise_datasets(yaml_config: Path) -> Dict[str, None]:
    """Add keys to dict that will contain all datasets defined in the YAML
    config.

    Parameters
    ----------
    yaml_config : Path
        The file containing the processing pipeline info as YAML

    Returns
    -------
    Dict
        The dict of datasets, whose keys are the names of the datasets, and
        values will eventually be arrays (but initialised to None in this
        function)
    """
    datasets = {}
    # Define the params related to dataset names in the given function, whether
    # it's a loader or a method function
    loader_dataset_params = ['name']
    method_dataset_params = ['data_in', 'data_out']

    yaml_conf = open_yaml_config(yaml_config)
    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        if 'loaders' in module_name:
            dataset_params = loader_dataset_params
        else:
            dataset_params = method_dataset_params

        # Add dataset param value to the dict of datasets if it doesn't already
        # exist
        _, method_conf = module_conf.popitem()
        for dataset_param in dataset_params:
            if method_conf[dataset_param] not in datasets:
                datasets[method_conf[dataset_param]] = None

    return datasets


def _get_method_funcs(yaml_config: Path) -> List[Tuple[str, Callable, Dict, bool]]:
    """Gather all the python functions needed to run the defined processing
    pipeline.

    Parameters
    ----------
    yaml_config : Path
        The file containing the processing pipeline info as YAML

    Returns
    -------
    List[Tuple[Callable, Dict, bool]]
        A list, each element being a tuple containing four elements:
        - a package name
        - a method function
        - a dict of parameters for the method function
        - a boolean describing if it is a loader function or not
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
            module = import_module(module_name)
            method_name, method_conf = module_conf.popitem()
            method_conf['method_name'] = method_name
            method_func = getattr(module, method_name)
            method_funcs.append((
                split_module_name[0],
                method_func,
                method_conf,
                is_loader
            ))
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
            module_name = '.'.join(split_module_name[:-1])
            wrapper_module_name = f"wrappers.{module_name}"
            wrapper_module = import_module(wrapper_module_name)
            wrapper_func_name = split_module_name[-1]
            wrapper_func = getattr(wrapper_module, wrapper_func_name)
            method_name, method_conf = module_conf.popitem()
            method_conf['method_name'] = method_name
            method_funcs.append((
                split_module_name[0],
                wrapper_func,
                method_conf,
                False
            ))
        else:
            err_str = f"An unknown module name was encountered: " \
                      f"{split_module_name[0]}"
            raise ValueError(err_str)

    return method_funcs


def _run_loader(func: Callable, params: Dict) -> Tuple[ndarray, ndarray,
                                                       ndarray, ndarray,
                                                       ndarray, int, int, int]:
    """Run a loader function in the processing pipeline.

    Parameters
    ----------
    func : Callable
        The python function that performs the loading.
    params : Dict
        A dict of parameters for the loader.

    Returns
    -------
    Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int, int, int]
        A tuple of 8 values that all loader functions return.
    """
    return func(**params)


def _run_method(func: Callable, method_name:str, method_params: Dict,
                httomo_params: Dict) -> ndarray:
    """Run a method function in the processing pipeline.

    Parameters
    ----------
    func : Callable
        The python function that performs the method.
    method_name : str
        The name of the method to apply.
    method_params : Dict
        A dict of parameters for the tomopy method.
    httomo_params : Dict
        A dict of parameters related to HTTomo.

    Returns
    -------
    ndarray
        An array containing the result of the method function.
    """
    return func(method_params, method_name, **httomo_params)


def _check_signature_for_httomo_params(func: Callable,
                                       params: List[Tuple[List[str], object]]) -> Dict:
    """Check if the given method requires any parameters related to HTTomo.

    Parameters
    ----------
    func : Callable
        Function whose type signature is to be inspected

    params : List[Tuple[List[str], object]]
        Each tuples contains a parameter name and the associated value that
        should be added if a method requires that parameter (note: multiple
        parameter names can be given in case the parameter isn't consistently
        named across tomopy functions, such as "angles" vs "angles_radians")

    Returns
    -------
    Dict
        A dict with the parameter names and values to be added for the given
        method function
    """
    extra_params = {}
    sig_params = signature(func).parameters
    for names, val in params:
        for name in names:
            if name in sig_params:
                extra_params[name] = val
    return extra_params


def _check_method_params_for_datasets(params: Dict,
                                      datasets: Dict[str, ndarray]) -> Dict:
    """Check a given method function's parameter values to see if any of them
    are a dataset.

    Parameters
    ----------
    params : Dict
        The dict of param names and their values for a given method function.

    datasets: Dict[str, ndarray]
        The dict of dataset names and their associated arrays.

    Returns
    -------
    Dict
        A dict containing the parameter names assigned to particular datasets,
        and the associated dataset values. This dict can be used to update the
        method function's params dict to reflect the dataset arrays that the
        method function requires.
    """
    dataset_params = {}
    for name, val in params.items():
        # If the value of this parameter is a dataset, then replace the
        # parameter value with the dataset value
        if val in datasets:
            dataset_params[name] = datasets[val]
    return dataset_params
