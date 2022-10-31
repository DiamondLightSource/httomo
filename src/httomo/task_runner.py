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

from httomo.utils import print_once
from httomo.yaml_utils import open_yaml_config
from httomo.data.hdf._utils.save import intermediate_dataset
from httomo._stats.globals import min_max_mean_std


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

    # Define list to store dataset stats for each task in the user config YAML
    glob_stats = _initialise_stats(yaml_config)

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
        method_name = params.pop('method_name')
        print_once(f"Running task {idx+1}: {method_name}...", comm)
        if is_loader:
            params.update(loader_extra_params)
            data, flats, darks, angles, angles_total, detector_y, detector_x = \
                _run_loader(func, params)

            # Update `datasets` dict with the data that has been loaded by the
            # loader
            datasets[params['name']] = data
            datasets['flats'] = flats
            datasets['darks'] = darks

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

            # Get the information describing if the method is being run only
            # once, or multiple times with different input datasets
            if 'data_in' in params.keys() and 'data_out' in params.keys():
                data_in = params.pop('data_in')
                data_out = params.pop('data_out')
            elif 'data_in_multi' in params.keys() and \
                'data_out_multi' in params.keys():
                data_in = params.pop('data_in_multi')
                data_out = params.pop('data_out_multi')
            else:
                # TODO: This error reporting is possibly better handled by
                # schema validation of the user config YAML
                err_str = "Invalid in/out dataset parameters"
                raise ValueError(err_str)

            # Check if the method function's params require any datasets stored
            # in the `datasets` dict
            dataset_params = _check_method_params_for_datasets(params, datasets)
            # Update the relevant parameter values according to the required
            # datasets
            params.update(dataset_params)

            # Make the input datasets a list if it's not, just to be generic and
            # below loop through all the datasets that the method should be
            # applied to
            if type(data_in) is str and type(data_out) is str:
                data_in = [data_in]
                data_out = [data_out]

            # Check if method type signature requires global statistics
            req_glob_stats = 'glob_stats' in signature(func).parameters

            for in_dataset, out_dataset in zip(data_in, data_out):
                if save_result:
                    out_dir = run_out_dir
                else:
                    out_dir = None

                if req_glob_stats is True:
                    stats = _fetch_glob_stats(datasets[in_dataset], comm)
                    glob_stats[idx][in_dataset] = stats
                    params.update({'glob_stats': stats})

                _run_method(func, idx+1, package, method_name, in_dataset,
                            out_dataset, datasets, params, httomo_params,
                            comm, out_dir=out_dir)


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

    yaml_conf = open_yaml_config(yaml_config)
    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        _, method_conf = module_conf.popitem()
        if 'loaders' in module_name:
            dataset_params = loader_dataset_params
        else:
            if 'data_in_multi' in method_conf.keys() and \
                'data_out_multi' in method_conf.keys():
                method_dataset_params = ['data_in_multi', 'data_out_multi']
            else:
                method_dataset_params = ['data_in', 'data_out']
            dataset_params = method_dataset_params

        # Add dataset param value to the dict of datasets if it doesn't already
        # exist
        for dataset_param in dataset_params:
            # Check if there are multiple input/output datasets to account for
            if type(method_conf[dataset_param]) is list:
                for dataset_name in method_conf[dataset_param]:
                    if dataset_name not in datasets:
                        datasets[dataset_name] = None
            else:
                if method_conf[dataset_param] not in datasets:
                    datasets[method_conf[dataset_param]] = None

    return datasets


# TODO: There's a lot of overlap with the `_initialise_datasets()` function, the
# only difference is that this function doesn't need to inspect the output
# datasets of a method, so perhaps the two functions can be nicely merged?
def _initialise_stats(yaml_config: Path) -> List[Dict]:
    """Generate a list of dicts that will hold the stats for the datasets in all
    the methods in the pipeline.

    Parameters
    ----------
    yaml_config : Path
        The file containing the processing pipeline info as YAML

    Returns
    -------
    List[Dict]
        A list containing the stats of all datasets of all methods in the
        pipeline.
    """
    stats = []
    # Define the params related to dataset names in the given function, whether
    # it's a loader or a method function
    loader_dataset_param = 'name'

    yaml_conf = open_yaml_config(yaml_config)
    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        _, method_conf = module_conf.popitem()

        if 'loaders' in module_name:
            dataset_param = loader_dataset_param
        else:
            if 'data_in_multi' in method_conf.keys():
                method_dataset_param = 'data_in_multi'
            else:
                method_dataset_param = 'data_in'
            dataset_param = method_dataset_param

        # Dict to hold the stats for each dataset associated with the method
        method_stats = {}

        # Check if there are multiple input datasets to account for
        if type(method_conf[dataset_param]) is list:
            for dataset_name in method_conf[dataset_param]:
                method_stats[dataset_name] = None
        else:
            method_stats[method_conf[dataset_param]] = None

        stats.append(method_stats)

    return stats


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


def _run_method(func: Callable, task_no: int, package_name: str,
                method_name: str, in_dataset: str, out_dataset: str,
                datasets: Dict[str, ndarray], method_params: Dict,
                httomo_params: Dict, comm: MPI.Comm,
                out_dir: str=None) -> ndarray:
    """Run a method function in the processing pipeline.

    Parameters
    ----------
    func : Callable
        The python function that performs the method.
    task_no : int
        The number of the given task, starting at index 1.
    package_name : str
        The package that the method function `func` comes from.
    method_name : str
        The name of the method to apply.
    in_dataset : str
        The name of the input dataset.
    out_dataset : str
        The name of the output dataset.
    datasets : Dict[str, ndarray]
        A dict containing all available datasets in the given pipeline.
    method_params : Dict
        A dict of parameters for the method.
    httomo_params : Dict, optional
        A dict of parameters related to HTTomo.
    comm : MPI.Comm
        MPI communicator object.
    out_dir : str, optional
        If the result should be saved in an intermediate file, the directory to
        save it should be provided.

    Returns
    -------
    ndarray
        An array containing the result of the method function.
    """
    # Add the appropriate dataset to the method function's dict of
    # parameters based on the parameter name for the method's python
    # function
    if package_name == 'tomopy':
        httomo_params['data'] = datasets[in_dataset]
    elif package_name == 'httomo':
        data_param = _set_method_data_param(func, in_dataset, datasets)
        httomo_params.update(data_param)

    # Run the method, then store the result in the appropriate
    # dataset in the `datasets` dict
    if package_name == 'tomopy':
        datasets[out_dataset] = \
            _run_tomopy_method(func, method_name, method_params, httomo_params)
    elif package_name == 'httomo':
        method_params.update(httomo_params)
        datasets[out_dataset] = _run_httomo_method(func, method_params)

    # TODO: The dataset saving functionality only supports 3D data
    # currently, so check that the dimension of the data is 3 before
    # saving it
    is_3d = len(datasets[out_dataset].shape) == 3
    # Save the result if necessary
    if out_dir is not None and is_3d:
        intermediate_dataset(datasets[out_dataset], out_dir,
                            comm, task_no, package_name, method_name,
                            out_dataset,
                            recon_algorithm=method_params.pop('algorithm', None))


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


def _run_tomopy_method(func: Callable, method_name:str, method_params: Dict,
                       httomo_params: Dict) -> ndarray:
    """Run a tomopy method function in the processing pipeline.

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


def _run_httomo_method(func: Callable, params: Dict) -> ndarray:
    """Run an HTTomo method function in the processing pipeline.

    Parameters
    ----------
    func : Callable
        The python function that performs the method.
    params : Dict
        A dict of parameters.

    Returns
    -------
    ndarray
        An array containing the result of the method function.
    """
    return func(**params)


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


def _set_method_data_param(func: Callable, dataset_name: str,
                           datasets: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """Set a key in the param dict whose value is the array associated with the
    input dataset name given in the YAML config. The name of this key must be
    the same as the associated parameter name in the python function which
    performs the method, in order to pass the parameters via dict-unpacking.

    E.g. suppose a method function has parameters:

    def my_func(sino: ndarray, ...) -> ndarray:
        ...

    and in the user config YAML file, the dataset name to be passed into this
    method function is called `tomo`, defined by YAML like the following:
    ```
    my_func:
      data_in: tomo
      data_out: tomo_out
    ```

    This function `_set_method_data_param()` would return a dict that contains:
    - the key `sino` mapped to the value of the array associated to the `tomo`
      dataset

    Parameters
    ----------
    func : Callable
        The method function whose data parameters will be inspected.
    dataset_name : str
        The name of the input dataset name from the user config YAML.
    datasets : Dict[str, ndarray]
        A dict of all the available datasets in the current run.

    Returns
    -------
    Dict[str, ndarray]
        A dict containing the input data parameter key and value needed for the
        given method function.
    """
    sig_params = list(signature(func).parameters.keys())
    # For httomo functions, the input data paramater will always be the first
    # parameter
    data_param = sig_params[0]
    return {data_param: datasets[dataset_name]}


def _fetch_glob_stats(data: ndarray, comm: MPI.Comm) -> Tuple[float, float,
                                                              float, float]:
    """Fetch the mix, max, mean, standard deviation of the given data.

    Parameters
    ----------
    data : ndarray
        The data to calculate statistics from.
    comm : MPI.Comm
        MPI communicator object.

    Returns
    -------
    Tuple[float, float, float, float]
        A tuple containing the stats values.
    """
    return min_max_mean_std(data, comm)
