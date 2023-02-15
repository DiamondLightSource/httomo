import multiprocessing
from datetime import datetime
from os import mkdir
from pathlib import Path
from typing import List, Dict, Tuple
from collections.abc import Callable
from inspect import signature
from importlib import import_module

from numpy import ndarray
import cupy as cp
from mpi4py import MPI

from httomo.utils import print_once, Pattern, _get_slicing_dim
from httomo.yaml_utils import open_yaml_config
from httomo.data.hdf._utils.save import intermediate_dataset
from httomo.data.hdf._utils.reslice import reslice
from httomo.data.hdf._utils.chunk import save_dataset, get_data_shape
from httomo._stats.globals import min_max_mean_std


# TODO: Define a list of savers which have no output dataset and so need to
# be treated differently to other methods. Probably should be handled in a
# more robust way than this workaround.
SAVERS_NO_DATA_OUT_PARAM = ['save_to_images']

reslice_warn_str = f"WARNING: Reslicing is performed more than once in this " \
                   f"pipeline, is there a need for this?"
# Hardcoded string that is used to check if a method is in a reconstruction
# module or not
RECON_MODULE_MATCH = 'recon.algorithm'


def run_tasks(
    in_file: Path,
    yaml_config: Path,
    out_dir: Path,
    dimension: int,
    pad: int = 0,
    ncore: int = 1,
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
    pad : int
        The padding size to use. Defaults to 0.
    ncore : int
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
        ncore = multiprocessing.cpu_count() # use all available CPU cores if not an MPI run

    # GPU related MPI communicators and indices
    num_GPUs = cp.cuda.runtime.getDeviceCount()
    gpu_id = int(comm.rank / comm.size * num_GPUs)
    gpu_comm = comm.Split(gpu_id)
    proc_id = f"[{gpu_id}:{gpu_comm.rank}]"

    # Define dict to store arrays that are specified as datasets in the user
    # config YAML
    datasets = _initialise_datasets(yaml_config, SAVERS_NO_DATA_OUT_PARAM)

    # Define list to store dataset stats for each task in the user config YAML
    glob_stats = _initialise_stats(yaml_config)

    # Get a list of the python functions associated to the methods defined in
    # user config YAML
    method_funcs = _get_method_funcs(yaml_config)

    # Define dict of params that are needed by loader functions
    loader_extra_params = {
        'in_file': in_file,
        'dimension': dimension,
        'pad': pad,
        'comm': comm
    }

    # Hardcoded string that is used to check if a method is in a reconstruction
    # module or not
    RECON_MODULE_MATCH = 'recon.algorithm'
    # A counter to track how many reslices occur in the processing pipeline
    reslice_counter = 0
    has_reslice_warn_printed = False
    
    # get a list with booleans to identify when reslicing needed (True) or not
    # (False).
    patterns = [f.pattern for (_, f, _, _) in method_funcs]
    reslice_bool_list = _check_if_should_reslice(patterns)

    # Run the methods
    for idx, (module_path, func, params, is_loader) in enumerate(method_funcs):
        package = module_path.split('.')[0]
        method_name = params.pop('method_name')
        task_no_str = f"Running task {idx+1}"
        pattern_str = f"(pattern={func.pattern.name})"
        print_once(f"{task_no_str} {pattern_str}: {method_name}...", comm)
        if is_loader:
            params.update(loader_extra_params)

            # Check if a value for the `preview` parameter of the loader has
            # been provided
            if 'preview' not in params.keys():
                params['preview'] = [None]

            data, flats, darks, angles, angles_total, detector_y, detector_x = \
                _run_loader(func, params)

            # Update `datasets` dict with the data that has been loaded by the
            # loader
            datasets[params['name']] = data
            datasets['flats'] = flats
            datasets['darks'] = darks

            # Define all params relevant to httomo that a wrapper function might
            # need
            possible_extra_params = [
                (['darks'], darks),
                (['flats'], flats),
                (['angles', 'angles_radians'], angles),
                (['comm'], comm),
                (['gpu_id'], gpu_id),
                (['out_dir'], run_out_dir)
            ]
        else:
            # adding ncore argument into params
            params.update({'ncore': ncore})

            reslice_counter, has_reslice_warn_printed = \
                 _run_method(idx, save_all, module_path, package, method_name,
                             params, possible_extra_params, len(method_funcs),
                             method_funcs[idx][1], method_funcs[idx-1][1],
                             datasets, run_out_dir, glob_stats, comm,
                             reslice_counter, has_reslice_warn_printed,
                             reslice_bool_list)

    # Print the number of reslice operations peformed in the pipeline
    reslice_summary_str = f"Total number of reslices: {reslice_counter}"
    reslice_summary_colour = 'blue' if reslice_counter <= 1 else 'red'
    print_once(reslice_summary_str, comm=comm, colour=reslice_summary_colour)


def _initialise_datasets(yaml_config: Path,
                         savers_no_data_out_param: List[str]) -> Dict[str, None]:
    """Add keys to dict that will contain all datasets defined in the YAML
    config.

    Parameters
    ----------
    yaml_config : Path
        The file containing the processing pipeline info as YAML
    savers_no_data_out_param : List[str]
        A list of savers which have neither `data_out` nor `data_out_multi` as
        their output.

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
    # TODO: For now, make savers such as `save_to_images` a special case where:
    # - `data_in` is defined
    # - but `data_out` is not, since the output of the method is not a dataset
    # but something else, like a directory containing many subdirectories of
    # images.
    # And therefore, don't try to inspect the `data_out` parameter of the method
    # to then try and initialise a dataset from its value, since it won't exist!
    savers_no_data_out_params = ['data_in']

    yaml_conf = open_yaml_config(yaml_config)
    for task_conf in yaml_conf:
        module_name, module_conf = task_conf.popitem()
        method_name, method_conf = module_conf.popitem()
        if 'loaders' in module_name:
            dataset_params = loader_dataset_params
        elif method_name in savers_no_data_out_param:
            dataset_params = savers_no_data_out_params
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
                module_name,
                method_func,
                method_conf,
                is_loader
            ))
        elif (split_module_name[0] == 'tomopy') or (split_module_name[0] == 'httomolib'):
            # The structure of wrapper functions for tomopy and httomolib is that 
            # each module in tomopy/httomolib is represented by a function in httomo.
            #
            # For example, the module `tomopy.misc.corr` module (which contains
            # the `median_filter()` method function) is represented in httomo in
            # the `wrappers.tomopy.misc` module as the `corr()` function.
            #
            # Different method functions that are available in the
            # `tomopy.misc.corr` module are then exposed in httomo by passing a
            # `method_name` parameter to the corr() function via the YAML config
            # file.
            wrapper_module_name = '.'.join(split_module_name[:-1])
            wrapper_module_name = f"wrappers.{wrapper_module_name}"
            wrapper_module = import_module(wrapper_module_name)
            wrapper_func_name = split_module_name[-1]
            wrapper_func = getattr(wrapper_module, wrapper_func_name)
            method_name, method_conf = module_conf.popitem()
            method_conf['method_name'] = method_name
            method_funcs.append((
                module_name,
                wrapper_func,
                method_conf,
                False
            ))
        else:
            err_str = f"An unknown module name was encountered: " \
                      f"{split_module_name[0]}"
            raise ValueError(err_str)

    return method_funcs


def _run_method(task_idx: int, save_all: bool, module_path: str,
                package_name: str, method_name: str, params: Dict,
                misc_params: Dict, no_of_tasks: int, current_func: Callable,
                prev_func: Callable, datasets: Dict, out_dir: str,
                glob_stats: List, comm: MPI.Comm, reslice_counter: int,
                has_reslice_warn_printed: bool,
                reslice_bool_list: List[bool]) -> Tuple[bool, bool]:
    """Run a method function in the processing pipeline.

    Parameters
    ----------
    task_idx : int
        The index of the current task (zero-based indexing).
    save_all : bool
        Whether to save the result of all methods in the pipeline,
    module_path : str
        The path of the module that the method function comes from.
    package_name : str
        The name of the package that the method function comes from.
    method_name : str
        The name of the method.
    params : Dict
        A dict of parameters for the method.
    misc_params : Dict
        A list of possible extra params that may be needed by a method.
    no_of_tasks : int
        The number of tasks in the pipeline.
    current_func : Callable
        The python function that performs the method.
    prev_func : Callable
        The python function that performed the previous method in the pipeline.
    datasets : Dict
        A dict of all the datasets involved in the pipeline.
    out_dir : str
        The path to the output directory of the run.
    glob_stats : List
    comm : MPI.Comm
        The MPI communicator used for the run.
    reslice_counter : int
        A counter for how many times reslicing has occurred in the pipeline.
    has_reslice_warn_printed : bool
        A flag to describe if the reslice warning has been printed or not.
    reslice_bool_list : List[bool]
        A list of boolens to describe which methods need reslicing of their
        input data prior to running.

    Returns
    -------
    Tuple[int, bool]
        Contains the `reslice_counter` and `has_reslice_warn_printed` values to
        enable the information to persist across method executions.
    """
    save_result = _check_save_result(task_idx, no_of_tasks, module_path,
                                     save_all, params.pop('save_result', None))

    # Check if the input dataset should be resliced before the task runs
    should_reslice = reslice_bool_list[task_idx]
    if should_reslice:
        reslice_counter += 1
        current_slice_dim = _get_slicing_dim(prev_func.pattern)
        next_slice_dim = _get_slicing_dim(current_func.pattern)

    if reslice_counter > 1 and not has_reslice_warn_printed:
        print_once(reslice_warn_str, comm=comm, colour='red')
        has_reslice_warn_printed = True

    # Check for any extra params unrelated to tomopy but related to
    # HTTomo that should be added in
    httomo_params = \
        _check_signature_for_httomo_params(current_func, misc_params)

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
        if method_name not in SAVERS_NO_DATA_OUT_PARAM:
            err_str = "Invalid in/out dataset parameters"
            raise ValueError(err_str)
        else:
            data_in = [params.pop('data_in')]
            data_out = [None]

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
    req_glob_stats = 'glob_stats' in signature(current_func).parameters

    # Check if a parameter sweep is defined for any of the method's
    # parameters
    current_param_sweep = False
    for k, v in params.items():
        if type(v) is tuple:
            param_sweep_name = k
            param_sweep_vals = v
            current_param_sweep = True
            break

    # Create a list to store the result of the different parameter values
    if current_param_sweep:
        out = []

    for in_dataset, out_dataset in zip(data_in, data_out):
        if method_name in SAVERS_NO_DATA_OUT_PARAM:
            # Perform a reslice of the data if necessary
            if should_reslice:
                resliced_data, _ = reslice(datasets[in_dataset],
                                           out_dir, current_slice_dim,
                                           next_slice_dim, comm)
                datasets[in_dataset] = resliced_data

            # Add the appropriate dataset to the method function's dict of
            # parameters based on the parameter name for the method's python
            # function
            if package_name in ['httomolib', 'tomopy']:
                httomo_params['data'] = datasets[in_dataset]

            # Add global stats if necessary
            if req_glob_stats is True:
                stats = _fetch_glob_stats(datasets[in_dataset], comm)
                glob_stats[task_idx][in_dataset] = stats
                params.update({'glob_stats': stats})

            _run_method_wrapper(current_func, method_name, params,
                                httomo_params)
            # Nothing more to do if the saver has a special kind of output which
            # handles saving the result
            return reslice_counter, has_reslice_warn_printed
        else:
            if current_param_sweep:
                # TODO: Assumes that only one input and output dataset have been
                # specified when doing a parameter sweep
                if len(data_in) > 1 and len(data_out) > 1:
                    err_str = f'Parameter sweeps are only implemented for a ' \
                              f'single input/output dataset'
                    raise ValueError(err_str)

                # Perform a reslice of the data if necessary
                if should_reslice:
                    resliced_data, _ = reslice(datasets[in_dataset],
                                               out_dir, current_slice_dim,
                                               next_slice_dim, comm)
                    datasets[in_dataset] = resliced_data

                # Add the appropriate dataset to the method function's dict of
                # parameters based on the parameter name for the method's python
                # function
                if package_name in ['httomolib', 'tomopy']:
                    httomo_params['data'] = datasets[in_dataset]

                # Add global stats if necessary
                if req_glob_stats is True:
                    stats = _fetch_glob_stats(datasets[in_dataset], comm)
                    # TODO: The `glob_stats` dict is not yet implemented to
                    # contain the stats for the different arrays resulting from
                    # a parameter sweep
                    #glob_stats[task_idx][in_dataset] = stats
                    params.update({'glob_stats': stats})

                for val in param_sweep_vals:
                    params[param_sweep_name] = val
                    res = _run_method_wrapper(current_func, method_name, params,
                                              httomo_params)
                    out.append(res)
                datasets[data_out[0]] = out
            else:
                # If the data is a list of arrays, then it was the result of a
                # parameter sweep from a previous method, so the next method
                # must be applied to all arrays in the list
                if type(datasets[data_in[0]]) is list:
                    for i, arr in enumerate(datasets[data_in[0]]):
                        # Perform a reslice of the data if necessary
                        if should_reslice:
                            resliced_data, _ = reslice(datasets[data_in[0]][i],
                                                       out_dir,
                                                       current_slice_dim,
                                                       next_slice_dim, comm)
                            datasets[data_in[0]][i] = resliced_data
                            arr = resliced_data

                        httomo_params['data'] = arr

                        # TODO: Add global stats if necessary
                        if req_glob_stats is True:
                            err_str = f'Methods requiring global stats are ' \
                                      f'not yet implemented to run after a ' \
                                      f'parameter sweep.'
                            raise ValueError(err_str)

                        res = _run_method_wrapper(current_func, method_name,
                                                  params, httomo_params)
                        datasets[data_out[0]][i] = res
                else:
                    # Perform a reslice of the data if necessary
                    if should_reslice:
                        resliced_data, _ = reslice(datasets[in_dataset],
                                                   out_dir, current_slice_dim,
                                                   next_slice_dim, comm)
                        datasets[in_dataset] = resliced_data

                    # Add the appropriate dataset to the method function's dict
                    # of parameters based on the parameter name for the method's
                    # python function
                    if package_name in ['httomolib', 'tomopy']:
                        httomo_params['data'] = datasets[in_dataset]

                    # Add global stats if necessary
                    if req_glob_stats is True:
                        stats = _fetch_glob_stats(datasets[in_dataset], comm)
                        glob_stats[task_idx][in_dataset] = stats
                        params.update({'glob_stats': stats})

                    # Run the method, then return the result for storage in the
                    # appropriate dataset in the `datasets` dict
                    res = _run_method_wrapper(current_func, method_name, params,
                                              httomo_params)
                    datasets[out_dataset] = res

        print_once(method_name, comm)
        # TODO: The dataset saving functionality only supports 3D data
        # currently, so check that the dimension of the data is 3 before
        # saving it
        is_3d = len(res.shape) == 3
        # Save the result if necessary
        any_param_sweep = type(datasets[data_out[0]]) is list
        if save_result and is_3d and not any_param_sweep:
            intermediate_dataset(datasets[out_dataset], out_dir,
                                 comm, task_idx+1,
                                 package_name, method_name,
                                 out_dataset,
                                 recon_algorithm=params.pop('algorithm', None))
        elif save_result and any_param_sweep:
            # Save the result of each value in the parameter sweep as a
            # different dataset within the same hdf5 file
            param_sweep_datasets = datasets[data_out[0]]
            slice_dim = _get_slicing_dim(current_func.pattern)
            data_shape = get_data_shape(param_sweep_datasets[0], slice_dim -1)
            file_name = \
                f"{task_idx}-{package_name}-{method_name}-{data_out[0]}.h5"
            for i in range(len(param_sweep_datasets)):
                dataset_name = f"/data/param_sweep_{i}"
                save_dataset(out_dir, file_name, param_sweep_datasets[i],
                             slice_dim=slice_dim,
                             chunks=(1, data_shape[1], data_shape[2]),
                             path=dataset_name, comm=comm)

    return reslice_counter, has_reslice_warn_printed


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


def _run_method_wrapper(func: Callable, method_name:str, method_params: Dict,
                       httomo_params: Dict) -> ndarray:
    """Run a wrapper method function (httomolib/tomopy) in the processing pipeline.

    Parameters
    ----------
    func : Callable
        The python function that performs the method.
    method_name : str
        The name of the method to apply.
    method_params : Dict
        A dict of parameters for the tomopy method.
    httomo_params : Dict
        A dict of parameters related to httomo.

    Returns
    -------
    ndarray
        An array containing the result of the method function.
    """
    return func(method_params, method_name, **httomo_params)


def _check_save_result(task_idx: int, no_of_tasks: int, module_path: str,
                       save_all: bool, save_result_param: bool) -> bool:
    """Check if the result of the current method should be saved.

    Parameters
    ----------
    task_idx : int
        The index of the current task (zero-based indexing).
    no_of_tasks : int
        The number of tasks in the pipeline.
    module_path : str
        The path of the module that the method function comes from.
    save_all : bool
        Whether to save the result of all methods in the pipeline,
    save_result_param : bool
        The value of `save_result` for given method form the YAML (if not
        defined in the YAML, then this will have a value of `None`).

    Returns
    -------
    bool
        Whether or not to save the result of a method.

    """
    save_result = False

    # Default behaviour for saving datasets is to save the output of the last
    # task, and the output of reconstruction methods are always saved unless
    # specified otherwise.
    #
    # The default behaviour can be overridden in two ways:
    # 1. the flag `--save_all` which affects all tasks
    # 2. the method param `save_result` which affects individual tasks
    #
    # Here, enforce default behaviour.
    if task_idx == no_of_tasks - 1:
        save_result = True

    # Now, check if `--save_all` has been specified, as this can override
    # default behaviour
    if save_all:
        save_result = True

    # Now, check if it's a method from a reconstruction module
    if RECON_MODULE_MATCH in module_path:
        save_result = True

    # Finally, check if the `save_result` param was specified in the YAML
    # config, as it can override both default behaviour and the `--save_all`
    # flag
    if save_result_param is not None:
        save_result = save_result_param

    return save_result


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
    """Fetch the min, max, mean, standard deviation of the given data.

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


def _check_if_should_reslice(patterns: List[Pattern]) -> List[bool]:
    """Determine if the input dataset for the method functions in the pipeline
    should be resliced. Builds the list of booleans.

    Parameters
    ----------
    patterns : List[Pattern]
        List of the patterns associated with the python functions needed for the
        run.

    Returns
    -------
    List[bool]
        List with booleans which methods need reslicing (True) or not (False).
    """
    # ___________Rules for when and when-not to reslice the data___________
    # In order to reslice more accurately we need to know about all patterns in
    # the given pipeline. 
    # The general rules are the following:
    # 1. Reslice ONLY if the pattern changes from "projection" to "sinogram" or the other way around
    # 2. With Pattern.all present one needs to check patterns on the edges of 
    # the Pattern.all. 
    # For instance consider the following example (method - pattern):
    #      1. Normalise - projection
    #      2. Dezinger - all
    #      3. Phase retrieval - projection
    #      4. Median - all
    #      5. Centering - sinogram
    # In this case you DON'T reclice between 2 and 3 as 1 and 3 are the same pattern.
    # You reclice between 4 and 5 as the pattern between 3 and 5 does change.
    total_number_of_methods = len(patterns)
    reslice_bool_list = [False] * total_number_of_methods

    current_pattern = patterns[0]
    for x in range(total_number_of_methods):
         if ((patterns[x] != current_pattern) and (patterns[x] != Pattern.all)):
             # skipping "all" pattern and look for different pattern from the
             # current pattern
             current_pattern = patterns[x]
             reslice_bool_list[x] = True
    return reslice_bool_list
